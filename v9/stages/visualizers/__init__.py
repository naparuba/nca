"""
Initialisation du module de visualiseurs spécialisés par stage.
"""

from .stage1_visualizer import Stage1Visualizer
from .stage2_visualizer import Stage2Visualizer
from .stage3_visualizer import Stage3Visualizer
from .stage4_visualizer import Stage4Visualizer

# Nouveaux composants de visualisation migrés
from .progressive_visualizer import ProgressiveVisualizer
from .intensity_animator import IntensityAwareAnimator
from .metrics_plotter import VariableIntensityMetricsPlotter
from .visualization_suite import create_complete_visualization_suite

# Dictionnaire des visualiseurs disponibles par stage
STAGE_VISUALIZERS = {
    1: Stage1Visualizer,
    2: Stage2Visualizer,
    3: Stage3Visualizer,
    4: Stage4Visualizer,
}

def get_visualizer(stage_id: int):
    """
    Récupère le visualiseur spécialisé pour un stage donné.
    Retourne None si aucun visualiseur spécifique n'est disponible.
    """
    return STAGE_VISUALIZERS.get(stage_id)

# Export des composants principaux
__all__ = [
    'Stage1Visualizer', 'Stage2Visualizer', 'Stage3Visualizer', 'Stage4Visualizer',
    'ProgressiveVisualizer', 'IntensityAwareAnimator', 'VariableIntensityMetricsPlotter',
    'create_complete_visualization_suite', 'get_visualizer', 'STAGE_VISUALIZERS'
]
