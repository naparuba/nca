"""
Registre auto-découvert des stages disponibles.
Charge dynamiquement tous les stages depuis le filesystem sans couplage aux numéros.

Principe:
- Parcourt automatiquement le répertoire stages/
- Détecte et charge les modules qui contiennent une classe héritant de BaseStage
- Indexe par slug (nom unique) au lieu de numéro
- Permet l'ajout de nouveaux stages sans modification de code

Avantages:
- Aucun import explicite nécessaire
- Découplage total entre les stages
- Extensibilité maximale
"""

import importlib
from pathlib import Path
from typing import Dict, Type, List, Optional

from .base_stage import BaseStage


class StageAutoRegistry:
    """
    Registre qui auto-découvre les stages depuis le filesystem.
    
    Chaque stage est identifié par un slug unique (ex: 'no_obstacles')
    et non plus par un numéro arbitraire.
    
    Le registre scanne automatiquement tous les sous-répertoires de stages/
    et charge les classes qui héritent de BaseStage.
    """
    
    def __init__(self, stages_dir: Path = None):
        """
        Initialise le registre et lance l'auto-découverte.
        
        Args:
            stages_dir: Répertoire contenant les stages (défaut: répertoire courant)
        """
        if stages_dir is None:
            stages_dir = Path(__file__).parent
        
        self.stages_dir = stages_dir
        self.device = "cpu"  # Device par défaut pour l'instanciation temporaire
        self._stages: Dict[str, Type[BaseStage]] = {}
        self._discover_stages()
    
    def _discover_stages(self):
        """
        Parcourt le répertoire stages/ et charge automatiquement
        tous les modules qui contiennent une classe Stage.
        
        Convention:
        - Chaque stage est dans son propre répertoire (ex: no_obstacles/)
        - Le module train.py contient la classe principale du stage
        - La classe doit hériter de BaseStage
        - Le slug est extrait de stage.config.name
        """
        print(f"\n🔍 Auto-découverte des stages dans {self.stages_dir}...")
        
        for stage_path in self.stages_dir.iterdir():
            # Ignore les fichiers et répertoires spéciaux
            if not stage_path.is_dir():
                continue
            
            if stage_path.name.startswith('_') or stage_path.name in ['visualizers', '__pycache__']:
                continue
            
            # Tentative de chargement du module train.py
            train_module_path = stage_path / 'train.py'
            if not train_module_path.exists():
                print(f"  ⚠️  {stage_path.name}/ ignoré (pas de train.py)")
                continue
            
            try:
                # Import dynamique du module
                # Format: stages.no_obstacles.train
                module_name = f"stages.{stage_path.name}.train"
                module = importlib.import_module(module_name)
                
                # Recherche de la classe Stage dans le module
                stage_class = self._find_stage_class(module)
                
                if stage_class:
                    # Création d'une instance temporaire pour récupérer le slug
                    # Le slug est défini dans stage.config.name
                    temp_instance = stage_class(device=self.device)
                    slug = temp_instance.config.name
                    
                    # Enregistrement dans le registre
                    self._stages[slug] = stage_class
                    print(f"  ✅ Stage '{slug}' découvert ({stage_class.__name__})")
                else:
                    print(f"  ⚠️  {stage_path.name}/ ignoré (pas de classe BaseStage trouvée)")
            
            except Exception as e:
                print(f"  ❌ Erreur lors du chargement de {stage_path.name}/: {e}")
                # On ne lève pas d'exception pour permettre le chargement partiel
                # Si un stage est cassé, les autres peuvent quand même fonctionner
    
    def _find_stage_class(self, module) -> Optional[Type[BaseStage]]:
        """
        Trouve la classe qui hérite de BaseStage dans un module.
        
        Parcourt tous les attributs du module et retourne la première classe
        qui hérite de BaseStage (en excluant BaseStage elle-même).
        
        Args:
            module: Module Python à analyser
            
        Returns:
            Classe trouvée ou None
        """
        for attr_name in dir(module):
            # Ignore les attributs privés
            if attr_name.startswith('_'):
                continue
            
            attr = getattr(module, attr_name)
            
            # Vérification que c'est une classe héritant de BaseStage
            # Conditions:
            # 1. C'est un type (une classe)
            # 2. C'est une sous-classe de BaseStage
            # 3. Ce n'est pas BaseStage elle-même (importée dans le module)
            if (isinstance(attr, type) and
                issubclass(attr, BaseStage) and
                attr is not BaseStage):
                return attr
        
        return None
    
    def get_stage(self, slug: str) -> Type[BaseStage]:
        """
        Récupère une classe de stage par son slug.
        
        Args:
            slug: Identifiant unique du stage (ex: 'no_obstacles')
            
        Returns:
            Classe du stage
            
        Raises:
            ValueError: Si le slug n'existe pas dans le registre
        """
        if slug not in self._stages:
            available = ', '.join(self._stages.keys())
            raise ValueError(
                f"Stage '{slug}' non trouvé dans le registre.\n"
                f"Stages disponibles: {available}"
            )
        
        return self._stages[slug]
    
    def list_stages(self) -> List[str]:
        """
        Liste tous les slugs de stages disponibles.
        
        Returns:
            Liste des slugs dans l'ordre alphabétique
        """
        return sorted(self._stages.keys())
    
    def create_stage(self, slug: str, device: str = "cpu", **kwargs) -> BaseStage:
        """
        Crée une instance d'un stage par son slug.
        
        Args:
            slug: Identifiant unique du stage
            device: Device PyTorch ('cpu' ou 'cuda')
            **kwargs: Arguments supplémentaires passés au constructeur du stage
            
        Returns:
            Instance du stage créée
            
        Raises:
            ValueError: Si le slug n'existe pas
        """
        stage_class = self.get_stage(slug)
        return stage_class(device=device, **kwargs)
    
    def has_stage(self, slug: str) -> bool:
        """
        Vérifie si un stage existe dans le registre.
        
        Args:
            slug: Identifiant du stage
            
        Returns:
            True si le stage existe
        """
        return slug in self._stages
    
    def get_stage_count(self) -> int:
        """
        Retourne le nombre de stages enregistrés.
        
        Returns:
            Nombre de stages
        """
        return len(self._stages)
