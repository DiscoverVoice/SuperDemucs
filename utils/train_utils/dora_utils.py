from dora import Dora

# Dora 실험 설정
experiment = Dora(experiment_name="audio_separation_experiment")
experiment.log_params({
    "dataset": config['dataset'],
    "model": config['model'],
    "best_params": study.best_params
})

# 실험 결과 로깅
experiment.log_metrics({
    "best_val_loss": study.best_value
})

# 모델 저장 경로 로깅
experiment.log_artifact('audio_separation_model.pt')
