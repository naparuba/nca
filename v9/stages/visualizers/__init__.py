"""
Initialisation du module de visualiseurs spécialisés par stage.
"""

from .stage1_visualizer import Stage1Visualizer
from .stage2_visualizer import Stage2Visualizer
from .stage3_visualizer import Stage3Visualizer
from .stage4_visualizer import Stage4Visualizer

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
