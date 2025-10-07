"""
Système d'auto-découverte des visualiseurs par stage.

REFACTORING v11:
- Auto-découverte des visualiseurs depuis chaque stage
- Plus de mapping hard-codé par numéro
- Plus de __all__ inutile
- Identification par slug uniquement

Les visualiseurs sont maintenant dans leurs répertoires de stages respectifs
et sont chargés dynamiquement.
"""

import importlib
from pathlib import Path
from typing import Optional, Type, Dict

# Imports des composants de visualisation avancés
# (ceux-ci ne sont pas spécifiques aux stages)
from .progressive_visualizer import ProgressiveVisualizer
from .intensity_animator import IntensityAwareAnimator
from .metrics_plotter import VariableIntensityMetricsPlotter
from .visualization_suite import create_complete_visualization_suite

class VisualizerRegistry:
    """
    Registre qui auto-découvre les visualiseurs depuis le filesystem.
    
    Chaque stage peut avoir son propre visualiseur dans visualizer.py.
    Le registre charge automatiquement tous les visualiseurs disponibles.
    """
    
    def __init__(self):
        self._visualizers: Dict[str, Type] = {}
        self._discover_visualizers()
    
    def _discover_visualizers(self):
        """
        Parcourt les répertoires de stages et charge leurs visualiseurs.
        """
        print(f"\n🔍 Auto-découverte des visualiseurs...")
        
        stages_dir = Path(__file__).parent.parent
        
        for stage_path in stages_dir.iterdir():
            # Ignore les fichiers et répertoires spéciaux
            if not stage_path.is_dir():
                continue
            
            if stage_path.name.startswith('_') or stage_path.name in ['visualizers', '__pycache__']:
                continue
            
            # Tentative de chargement du visualizer.py
            visualizer_path = stage_path / 'visualizer.py'
            if not visualizer_path.exists():
                continue
            
            try:
                # Import dynamique du module
                module_name = f"stages.{stage_path.name}.visualizer"
                module = importlib.import_module(module_name)
                
                # Recherche de la classe Visualizer dans le module
                visualizer_class = self._find_visualizer_class(module)
                
                if visualizer_class:
                    # Le slug est le nom du répertoire
                    slug = stage_path.name
                    self._visualizers[slug] = visualizer_class
                    print(f"  ✅ Visualiseur '{slug}' découvert ({visualizer_class.__name__})")
            
            except Exception as e:
                print(f"  ⚠️  Impossible de charger visualiseur {stage_path.name}: {e}")
    
    def _find_visualizer_class(self, module) -> Optional[Type]:
        """
        Trouve la classe de visualisation dans un module.
        Convention: classe se terminant par 'Visualizer'.
        """
        for attr_name in dir(module):
            if attr_name.startswith('_'):
                continue
            
            attr = getattr(module, attr_name)
            
            # Recherche des classes se terminant par 'Visualizer'
            if isinstance(attr, type) and attr_name.endswith('Visualizer'):
                return attr
        
        return None
    
    def get_visualizer(self, slug: str) -> Optional[Type]:
        """
        Récupère le visualiseur pour un stage par son slug.
        
        Args:
            slug: Identifiant du stage
            
        Returns:
            Classe du visualiseur ou None si non trouvé
        """
        if slug not in self._visualizers:
            print(f"  ⚠️  Aucun visualiseur trouvé pour le stage '{slug}'")
            return None
        
        return self._visualizers[slug]
    
    def has_visualizer(self, slug: str) -> bool:
        """Vérifie si un visualiseur existe pour un stage."""
        return slug in self._visualizers
    
    def list_visualizers(self) -> list:
        """Liste tous les slugs de visualiseurs disponibles."""
        return sorted(self._visualizers.keys())


# Instance globale du registre
_visualizer_registry = None


def get_visualizer(slug: str):
    """
    Récupère et instancie le visualiseur pour un stage donné.
    
    Args:
        slug: Identifiant du stage (ex: 'no_obstacles')
        
    Returns:
        Instance du visualiseur ou None
    """
    global _visualizer_registry
    
    if _visualizer_registry is None:
        _visualizer_registry = VisualizerRegistry()
    
    visualizer_class = _visualizer_registry.get_visualizer(slug)
    
    if visualizer_class:
        instance = visualizer_class()
        print(f"  ✅ Visualiseur pour stage '{slug}' instancié: {instance.__class__.__name__}")
        return instance
    
    return None

