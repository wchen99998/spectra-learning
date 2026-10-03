"""Native encoder interventions scored by fixed, saved chemical token readouts.

These are causal effects on a frozen decoder, not intrinsic chemical logits of
the encoder. Validation selects components; separate test rows confirm them.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import modal

from scripts.modal_encoder_chemistry import (LOCAL, PROJECT, ROOT, batch_inputs,
    error_metrics, image, metadata, provenance, restore, sha256, volume, write_json)

app = modal.App("spectra-native-chemistry-causal")


def selected_rows(data, fold, size):
    import numpy as np
    available = np.flatnonzero(data["fold"] == fold)
    if size > len(available):
        raise ValueError("Intervention cohort exceeds available molecules")
    return np.sort(np.random.default_rng(20260930).choice(available, size, replace=False))


def load_endpoints(name):
    import jax
    import jax.numpy as jnp
    from scripts.encoder_chemistry_readouts import load_readout
    directory = ROOT / name / "readouts/depth27/attention"
    extraction_path = ROOT / name / "extraction.json"
    extraction = json.loads(extraction_path.read_text())
    if not extraction["complete"] or not extraction["extraction_vs_native_pilot"]["bitwise_equal"]:
        raise ValueError("Exact native token extraction is required")
    params, records = [], []
    for seed in (0, 1, 2):
        p, info = load_readout(directory / f"seed{seed}")
        contract = info["contract"]
        if (contract["encoder"] != name or contract["token_scope"] != "physical_peaks_only"
            or contract["depth"] != 27 or info["kind"] != "attention" or info["seed"] != seed):
            raise ValueError("Causal decoder differs from the prespecified encoder/token scope")
        if (contract["extraction_sha256"] != sha256(extraction_path)
            or contract["token_source_sha256"] != extraction["files"]["tokens_bf16"]["sha256"]
            or contract["panel_sha256"]["inputs.npz"] != sha256(ROOT / "panel/inputs.npz")
            or contract["panel_sha256"]["labels.npz"] != sha256(ROOT / "panel/labels.npz")
            or contract["source_sha256"]["encoder_chemistry_readouts.py"] != sha256(PROJECT / "scripts/encoder_chemistry_readouts.py")):
            raise ValueError("Causal readout data/code contract differs from current artifacts")
        params.append(p)
        records.append(info)
    return jax.tree.map(lambda *x: jnp.stack(x), *params), records


def make_functions(schema):
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from scripts.encoder_chemistry_interventions import trace_encoder
    from scripts.encoder_chemistry_readouts import apply_readout

    @nnx.jit
    def forward(encoder, inputs, controls):
        return trace_encoder(encoder, inputs["peak_mz"], inputs["peak_intensity"],
            valid_mask=inputs["peak_valid_mask"], visible_mask=inputs["peak_valid_mask"],
            spectrum_metadata=metadata(inputs, schema), interventions=controls)

    @jax.jit
    def score(params, tokens, valid):
        mask = jnp.concatenate([valid, jnp.zeros((len(valid), 1), jnp.bool_)], axis=1)
        return jax.vmap(lambda p: apply_readout(p, tokens, mask, "attention")[0])(params)
    return forward, score


def component_variants(controls, head_layers=(), branch_layers=None):
    import jax.numpy as jnp
    variants = [("identity", controls)]
    for branch, field in (("attention", "attention_scale"), ("mlp", "mlp_scale"), ("metadata", "block_condition_scale")):
        layers = range(1, len(controls.attention_scale)+1) if branch_layers is None else branch_layers[branch]
        for number in layers:
            layer = number-1
            variants.append((f"{branch}_block{layer+1}_scale0",
                controls._replace(**{field: getattr(controls, field).at[layer].set(0)})))
    for dose in (0., .25, .5, .75):
        variants.append((f"final_metadata_scale{dose:g}", controls._replace(
            final_modulation_scale=jnp.asarray(dose, jnp.float32))))
    variants.append(("all_block_metadata_scale0", controls._replace(
        block_condition_scale=jnp.zeros_like(controls.block_condition_scale))))
    for layer in head_layers:
        for head in range(controls.head_scale.shape[1]):
            variants.append((f"head_block{layer}_head{head}_scale0", controls._replace(
                head_scale=controls.head_scale.at[layer-1, head].set(0))))
    return variants


@app.function(image=image, gpu="H100", cpu=8, memory=65536, timeout=14400,
              volumes={"/results": volume})
def components(name: str, fold: str = "val", size: int = 256, head_layers: tuple[int, ...] = (), selection: dict | None = None) -> dict:
    os.chdir(PROJECT)
    import jax
    import ml_dtypes
    import numpy as np
    from scripts.encoder_chemistry_interventions import identity_interventions

    jax.config.update("jax_default_matmul_precision", "highest")
    params, endpoints = load_endpoints(name)
    encoder, config = restore(name)
    data = np.load(ROOT / "panel/inputs.npz")
    rows = selected_rows(data, fold, size)
    batch_size = 8
    controls = identity_interventions(encoder, batch_size, 127)
    if selection is not None and (fold != "test" or selection["test_used"] or selection["encoder"] != name):
        raise ValueError("Confirmatory component selection must be frozen on validation for this encoder")
    variants = component_variants(controls, head_layers, None if selection is None else selection["branch_layers"])
    forward, score = make_functions(config["encoder_metadata_schema"])
    scores = np.empty((len(variants), len(rows), 3, 188), np.float32)
    geometry = np.empty((len(variants), len(rows)), np.float32)
    cached = np.load(ROOT / name / "tokens_bf16.npy", mmap_mode="r")
    baselines, parity = [], []
    started = time.monotonic()
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start+batch_size]
        inputs = batch_inputs(data, batch_rows, batch_size)
        tokens = forward(encoder, inputs, controls).final
        baselines.append(tokens)
        target = cached[-1, batch_rows].view(ml_dtypes.bfloat16)
        valid = np.concatenate([inputs["peak_valid_mask"][:len(batch_rows)], np.ones((len(batch_rows),1),bool)],1)
        parity.append(error_metrics(np.asarray(tokens)[:len(batch_rows)], target, valid))
    if not all(p["bitwise_equal"] for p in parity):
        raise ValueError("Causal forward differs from the fitted readout cache beyond the numerical gate")
    for index, (label, control) in enumerate(variants):
        for batch_index, start in enumerate(range(0, len(rows), batch_size)):
            batch_rows = rows[start:start+batch_size]
            inputs = batch_inputs(data, batch_rows, batch_size)
            tokens = baselines[batch_index] if index == 0 else forward(encoder, inputs, control).final
            logits = score(params, tokens, inputs["peak_valid_mask"])
            scores[index, start:start+len(batch_rows)] = np.asarray(logits)[:, :len(batch_rows)].transpose(1,0,2)
            mask = inputs["peak_valid_mask"][..., None]
            difference = (np.asarray(tokens[:,:-1], np.float32)-np.asarray(baselines[batch_index][:,:-1], np.float32))*mask
            denominator = np.square(np.asarray(baselines[batch_index][:,:-1],np.float32)*mask).sum((1,2))
            geometry[index,start:start+len(batch_rows)] = np.sqrt((difference**2).sum((1,2))/np.maximum(denominator,1e-20))[:len(batch_rows)]
        if index % 10 == 0 or index == len(variants)-1:
            print(name,fold,label,index+1,len(variants),round(time.monotonic()-started,1),flush=True)
    suffix = "_selected" if selection is not None else ("_heads" + "-".join(map(str, head_layers)) if head_layers else "")
    output = ROOT / name / "causal" / f"components_{fold}_{size}{suffix}"
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output / "predictions.npz",logits=scores,token_relative_l2=geometry,rows=rows,
                        variants=np.asarray([v[0] for v in variants]))
    record = {**provenance(name,config),"fold":fold,"rows":rows.tolist(),"complete":True,
        "variants":[v[0] for v in variants],"endpoints":[p["weights_sha256"] for p in endpoints],
        "head_layers":list(head_layers),"selection":selection,"baseline_cache_parity":parity,
        "scope":"Effects on three fixed physical-token attentive chemical decoders; ablation is an artificial intervention, not molecule editing.",
        "conditioning_ablation":"Zero sample-dependent condition entering block affine; learned affine bias remains.",
        "seconds":time.monotonic()-started,"predictions_sha256":sha256(output/"predictions.npz")}
    write_json(output / "manifest.json",record)
    volume.commit()
    return record


@app.function(image=image, gpu="H100", cpu=8, memory=65536, timeout=14400,
              volumes={"/results": volume})
def patching(name: str, fold: str = "test", size: int = 256) -> dict:
    os.chdir(PROJECT)
    import jax
    import jax.numpy as jnp
    import numpy as np
    import ml_dtypes
    from scripts.encoder_chemistry_interventions import identity_interventions
    from scripts.encoder_chemistry_corruptions import corrupt_inputs

    jax.config.update("jax_default_matmul_precision", "highest")
    params,endpoints = load_endpoints(name)
    encoder,config = restore(name)
    data = np.load(ROOT / "panel/inputs.npz")
    rows = selected_rows(data,fold,size)
    corrupt, selected, corruption_record = corrupt_inputs(data,rows,name="attenuate_top20",seed=20260930)
    batch_size=8
    controls=identity_interventions(encoder,batch_size,127)
    forward,score = make_functions(config["encoder_metadata_schema"])
    labels=["clean","corrupt"]+[f"restore_{kind}_depth{depth}" for depth in range(26) for kind in ("selected","random","all")]
    logits=np.empty((len(labels),size,3,188),np.float32)
    started=time.monotonic()
    rng=np.random.default_rng(20260930)
    cache=np.load(ROOT/name/"tokens_bf16.npy",mmap_mode="r")
    cache_parity=[]
    for start in range(0,size,batch_size):
        ids=rows[start:start+batch_size]
        length=len(ids)
        clean_batch=batch_inputs(data,ids,batch_size)
        corrupt_batch=batch_inputs(corrupt,np.arange(start,start+length),batch_size)
        clean_trace=forward(encoder,clean_batch,controls)
        raw,clean_tokens=clean_trace.raw,clean_trace.final
        valid=np.concatenate([clean_batch["peak_valid_mask"][:length],np.ones((length,1),bool)],1)
        parity=error_metrics(np.asarray(clean_tokens)[:length],cache[-1,ids].view(ml_dtypes.bfloat16),valid)
        cache_parity.append(parity)
        if not parity["bitwise_equal"]:
            raise ValueError("Patching baseline is not bitwise equal to the fitted readout cache")
        corrupt_tokens=forward(encoder,corrupt_batch,controls).final
        target_mask=np.zeros((batch_size,128),bool)
        target_mask[:length,:-1]=selected[start:start+length]
        random_mask=np.zeros_like(target_mask)
        for i in range(length):
            available=np.flatnonzero(clean_batch["peak_valid_mask"][i] & ~target_mask[i,:-1])
            random_mask[i,rng.choice(available,size=int(target_mask[i].sum()),replace=False)]=True
        all_mask=np.concatenate([clean_batch["peak_valid_mask"],np.ones((batch_size,1),bool)],1)
        logits[0,start:start+length]=np.asarray(score(params,clean_tokens,clean_batch["peak_valid_mask"]))[:,:length].transpose(1,0,2)
        logits[1,start:start+length]=np.asarray(score(params,corrupt_tokens,clean_batch["peak_valid_mask"]))[:,:length].transpose(1,0,2)
        for depth in range(26):
            for k,mask in enumerate((target_mask,random_mask,all_mask)):
                intervention=controls._replace(patch_layer=jnp.asarray(depth,jnp.int32),patch_values=raw[depth],patch_mask=jnp.asarray(mask))
                tokens=forward(encoder,corrupt_batch,intervention).final
                logits[2+depth*3+k,start:start+length]=np.asarray(score(params,tokens,clean_batch["peak_valid_mask"]))[:,:length].transpose(1,0,2)
        print(name,fold,"patching",start+length,size,round(time.monotonic()-started,1),flush=True)
    output=ROOT/name/"causal"/f"patching_{fold}_{size}"
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/"predictions.npz",logits=logits,rows=rows,variants=np.asarray(labels),target_mask=selected)
    full_restore_max_abs=float(np.max(np.abs(logits[np.arange(4,len(labels),3)]-logits[0])))
    record={**provenance(name,config),"complete":True,"fold":fold,"rows":rows.tolist(),"variants":labels,
        "corruption":corruption_record,"endpoints":[p["weights_sha256"] for p in endpoints],
        "baseline_cache_parity":cache_parity,"full_state_rescue_max_abs_logit_error":full_restore_max_abs,
        "scope":"Selected physical-slot activation rescue versus same-count uncorrupted random slots; full-state restoration is the sham/numerical control.",
        "seconds":time.monotonic()-started,"predictions_sha256":sha256(output/"predictions.npz")}
    write_json(output/"manifest.json",record)
    volume.commit()
    return record


@app.local_entrypoint()
def main(name: str = "both", phase: str = "components", fold: str = "val", size: int = 256, head_layers: str = "", selection_path: str = ""):
    names=("parent_1190k","periodic_1209k") if name=="both" else (name,)
    function={"components":components,"patching":patching}[phase]
    heads=tuple(int(x) for x in head_layers.split(",") if x)
    selection=json.loads(Path(selection_path).read_text()) if selection_path else None
    if selection is not None:
        if name == "both" or phase != "components" or fold != "test":
            raise ValueError("A validation selection belongs to one encoder's test components")
        heads=tuple(selection["branch_layers"]["attention"])
    if heads and phase!="components":
        raise ValueError("Head selection applies only to component experiments")
    calls={n:function.spawn(n,fold,size,heads,selection) if phase=="components" else function.spawn(n,fold,size) for n in names}
    for n,call in calls.items():
        result=call.get()
        suffix="_selected" if selection is not None else ("_heads"+"-".join(map(str,heads)) if heads else "")
        run_name=f"{phase}_{fold}_{size}{suffix}"
        destination=LOCAL/n/"causal"/run_name
        write_json(destination/"manifest.json",result)
        remote=str((ROOT/n/"causal"/run_name/"predictions.npz").relative_to("/results"))
        with (destination/"predictions.npz").open("wb") as stream:
            for chunk in volume.read_file(remote):
                stream.write(chunk)
        if sha256(destination/"predictions.npz")!=result["predictions_sha256"]:
            raise ValueError("Downloaded intervention predictions differ from recorded hash")
