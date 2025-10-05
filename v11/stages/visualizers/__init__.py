"""
Initialisation du module de visualiseurs spécialisés par stage.
Les visualiseurs sont maintenant dans leurs répertoires de stages respectifs.
"""

from ..stage1 import Stage1Visualizer
from ..stage2 import Stage2Visualizer
from ..stage3 import Stage3Visualizer
from ..stage4 import Stage4Visualizer

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
    Instancie et retourne le visualiseur approprié.
    """
    # Importation directe pour Stage5Visualizer
    if stage_id == 5 and 5 not in STAGE_VISUALIZERS:
        from ..stage5.visualizer import Stage5Visualizer
        STAGE_VISUALIZERS[5] = Stage5Visualizer
        print(f"✅ Stage5Visualizer importé avec succès: {Stage5Visualizer}")
    
    visualizer_class = STAGE_VISUALIZERS.get(stage_id)
    if visualizer_class:
        instance = visualizer_class()  # Instanciation de la classe
        print(f"✅ Visualiseur pour Stage {stage_id} instancié: {instance}")
        return instance
    else:
        print(f"❌ Aucun visualiseur trouvé pour le Stage {stage_id}")
        return None

# Export des composants principaux
__all__ = [
    'Stage1Visualizer', 'Stage2Visualizer', 'Stage3Visualizer', 'Stage4Visualizer',
    'ProgressiveVisualizer', 'IntensityAwareAnimator', 'VariableIntensityMetricsPlotter',
    'create_complete_visualization_suite', 'get_visualizer', 'STAGE_VISUALIZERS'
]
