from typing import Optional

class TrainingConfig:
    def __init__(self, tasa_aprendizaje=1.0, max_epocas=1000, error_min=0.01, verbose=True, random_seed=None):
        self.tasa_aprendizaje = tasa_aprendizaje
        self.max_epocas = max_epocas
        self.error_min = error_min
        self.verbose = verbose
        self.random_seed = random_seed
    
    def to_dict(self):
        return {
            'tasa_aprendizaje': self.tasa_aprendizaje,
            'max_epocas': self.max_epocas,
            'error_min': self.error_min,
            'verbose': self.verbose,
            'random_seed': self.random_seed
        }


DEFAULT_CONFIG = TrainingConfig()

FAST_CONFIG = TrainingConfig(
    tasa_aprendizaje=2.0,
    max_epocas=500,
    error_min=0.05
)

PRECISE_CONFIG = TrainingConfig(
    tasa_aprendizaje=0.5,
    max_epocas=2000,
    error_min=0.001
)

GATE_CONFIG = TrainingConfig(
    tasa_aprendizaje=1.0,
    max_epocas=1000,
    error_min=0.01,
    verbose=True
)

REGRESSION_CONFIG = TrainingConfig(
    tasa_aprendizaje=0.1,
    max_epocas=100000,
    error_min=1e-6,
    verbose=True
)


def get_config(config_name: str = "default") -> TrainingConfig:
    configs = {
        'default': DEFAULT_CONFIG,
        'fast': FAST_CONFIG,
        'precise': PRECISE_CONFIG,
        'gate': GATE_CONFIG,
        'regression': REGRESSION_CONFIG
    }
    
    if config_name not in configs:
        available = ', '.join(configs.keys())
        raise ValueError(f"Configuración '{config_name}' no disponible. Disponibles: {available}")
    
    return configs[config_name]


def create_custom_config(**kwargs) -> TrainingConfig:
    config_dict = DEFAULT_CONFIG.to_dict()
    config_dict.update(kwargs)
    return TrainingConfig(**config_dict)
