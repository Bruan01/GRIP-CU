"""Exercise effective trainer retention and narrowly pinned recovery exceptions."""
from pathlib import Path
import hashlib,json,sys
import pytest

HNG=Path(__file__).parents[1]
sys.path[:0]=[str(HNG/'scripts'),str(HNG/'src')]
from source_checkpoint_retention import AMENDMENT,retention_limit,cli_retention,audit_retention,execution_contract,sha
from hard_negative_grip.run_protection import checkpoint_training_kwargs


@pytest.mark.parametrize('steps,interval,limit',[(32,8,5),(125,25,6)])
def test_real_trainer_mapping_retains_all_planned_states(steps,interval,limit):
    d=dict(optimizer_steps=steps,save_steps=interval)
    assert checkpoint_training_kwargs(interval,0)['save_total_limit']==1
    assert retention_limit(d)==limit
    assert checkpoint_training_kwargs(interval,limit)['save_total_limit']>=steps//interval


def test_bad_effective_retention_fails_before_gpu(tmp_path):
    c=dict(design=dict(optimizer_steps=32,save_steps=8))
    with pytest.raises(ValueError,match='effective checkpoint retention'):
        cli_retention(tmp_path,c,'frequency_matched',dict(save_steps=8,save_total_limit=0))
    cli_retention(tmp_path,c,'frequency_matched',dict(save_steps=8,save_total_limit=5))


def recovery_fixture(run):
    c=dict(design=dict(dataset='nell23k',optimizer_steps=32,scheduler_max_steps=240,save_steps=8),
           training_argv={name:['python','train','--save_total_limit','0','--stop_after_step','32'] for name in ['random9','frequency_matched']})
    (run/'contract.json').write_text(json.dumps(c))
    final=run/'random9/training/trainer_listed/checkpoint-32';final.mkdir(parents=True)
    for name in ['adapter_model.safetensors','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json']:(final/name).write_text('retained-final-state')
    adapter=run/'random9/training/listed/adapter';adapter.mkdir(parents=True)
    for name in ['adapter_model.safetensors','adapter_config.json']:(adapter/name).write_text('frozen-final-model')
    ctrl=run/'controller.py';ctrl.write_text('frozen controller');helper=run/'helper.py';helper.write_text('frozen helper')
    a=dict(schema='source-retention-amendment-v1',original_contract_sha256=sha(run/'contract.json'),
           controller_path=str(ctrl),controller_sha256=sha(ctrl),helper_path=str(helper),helper_sha256=sha(helper),
           original_argv_sha256={name:hashlib.sha256(json.dumps(argv,ensure_ascii=False).encode()).hexdigest() for name,argv in c['training_argv'].items()},
           retention_limits=dict(random9=0,frequency_matched=5),legacy_retention_loss=dict(random9=dict(missing_steps=[8,16,24],irrecoverable=True,
             final_adapter_sha256={name:sha(adapter/name) for name in ['adapter_model.safetensors','adapter_config.json']},checkpoint32_state_sha256=sha(final/'trainer_state.json'))))
    path=run/AMENDMENT;path.parent.mkdir(parents=True);path.write_text(json.dumps(a));path.with_suffix('.sha256').write_text(sha(path))
    return c,final,adapter


def test_explicit_missing_early_states_do_not_trigger_retraining_or_hide_loss(tmp_path):
    c,final,adapter=recovery_fixture(tmp_path)
    audit=audit_retention(tmp_path,c,'random9')
    assert audit['missing_steps']==[8,16,24] and audit['retained_steps']==[32] and audit['no_retraining']
    amended=execution_contract(tmp_path,c)
    assert c['training_argv']['frequency_matched'][3]=='0'
    assert amended['training_argv']['frequency_matched'][3]=='5'
    assert amended['training_argv']['random9']==c['training_argv']['random9']
    assert amended['training_argv']['frequency_matched'][4:]==c['training_argv']['frequency_matched'][4:]


@pytest.mark.parametrize('fault',['final_state','model','new_arm','policy','controller'])
def test_recovery_cannot_waive_new_loss_or_substitute_a_model(tmp_path,fault):
    c,final,adapter=recovery_fixture(tmp_path)
    if fault=='final_state':(final/'optimizer.pt').unlink()
    if fault=='model':(adapter/'adapter_model.safetensors').write_text('changed model')
    if fault=='policy':(tmp_path/AMENDMENT).write_text('{}')
    if fault=='controller':(tmp_path/'controller.py').write_text('changed controller')
    with pytest.raises(ValueError):audit_retention(tmp_path,c,'frequency_matched' if fault=='new_arm' else 'random9')
