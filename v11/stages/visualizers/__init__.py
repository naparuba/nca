"""
Initialisation du module de visualiseurs spécialisés par stage.
Les visualiseurs sont maintenant dans leurs répertoires de stages respectifs.
"""

from .intensity_animator import IntensityAwareAnimator
from .metrics_plotter import VariableIntensityMetricsPlotter
# Nouveaux composants de visualisation migrés
from .progressive_visualizer import ProgressiveVisualizer
from .visualization_suite import create_complete_visualization_suite



def get_visualizer(stage_id: int):
    from ..stage1 import Stage1Visualizer
    from ..stage2 import Stage2Visualizer
    from ..stage3 import Stage3Visualizer
    from ..stage4 import Stage4Visualizer
    from ..stage5 import Stage5Visualizer
    from ..stage6 import Stage6Visualizer
    
    # Dictionnaire des visualiseurs disponibles par stage
    STAGE_VISUALIZERS = {
        1: Stage1Visualizer,
        2: Stage2Visualizer,
        3: Stage3Visualizer,
        4: Stage4Visualizer,
        5: Stage5Visualizer,
        6: Stage6Visualizer,
        
    }
    """
    Récupère le visualiseur spécialisé pour un stage donné.
    Instancie et retourne le visualiseur approprié.
    """
    
    visualizer_class = STAGE_VISUALIZERS.get(stage_id)
    if not visualizer_class:
        raise (f"❌ Aucun visualiseur trouvé pour le Stage {stage_id}")
    
    instance = visualizer_class()  # Instanciation de la classe
    print(f"✅ Visualiseur pour Stage {stage_id} instancié: {instance}")
    return instance


# Export des composants principaux
__all__ = [
    'Stage1Visualizer', 'Stage2Visualizer', 'Stage3Visualizer', 'Stage4Visualizer',
    'ProgressiveVisualizer', 'IntensityAwareAnimator', 'VariableIntensityMetricsPlotter',
    'create_complete_visualization_suite', 'get_visualizer', 'STAGE_VISUALIZERS'
]
