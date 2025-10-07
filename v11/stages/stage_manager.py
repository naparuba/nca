"""
Gestionnaire de stages modulaire pour le NCA.
Orchestre l'exécution des stages de manière découplée et extensible.

REFACTORING v11:
- Utilise StageAutoRegistry pour l'auto-découverte des stages
- Utilise StageSequence pour définir l'ordre d'exécution
- Plus de couplage avec les numéros de stages
- Identification par slug uniquement
"""

from typing import Dict, Any, List, Optional
import json
from pathlib import Path

from .base_stage import BaseStage
from .registry import StageAutoRegistry
from .sequence import StageSequence


class ModularStageManager:
    """
    Gestionnaire principal des stages modulaires.
    Utilise maintenant le système de registry auto-découvert et la séquence configurable.
    """
    
    def __init__(self, global_config: Any, device: str = "cpu",
                 custom_sequence: Optional[List[str]] = None):
        """
        Initialise le gestionnaire de stages.
        
        Args:
            global_config: Configuration globale du système
            device: Device PyTorch ('cpu' ou 'cuda')
            custom_sequence: Séquence personnalisée de slugs (optionnel)
        """
        self.global_config = global_config
        self.device = device
        
        # 🎯 Nouveau système basé sur registry et séquence
        self.registry = StageAutoRegistry()
        self.sequence = StageSequence(custom_sequence)
        
        # Stages actifs indexés par SLUG, pas par numéro
        self.active_stages: Dict[str, BaseStage] = {}
        self.execution_history: List[Dict[str, Any]] = []
        
        # Configuration d'exécution
        self.total_epochs_planned = global_config.TOTAL_EPOCHS
        self.epochs_per_stage = self._calculate_epochs_per_stage()
    
    def _calculate_epochs_per_stage(self) -> Dict[str, int]:
        """
        Calcule le nombre d'époques par stage selon leur configuration.
        Utilise maintenant les SLUGS au lieu des numéros.
        """
        epochs_per_stage = {}
        
        print("\n=== Calcul détaillé de la répartition des époques par stage ===")
        print(f"Nombre total d'époques planifiées: {self.total_epochs_planned}")
        
        # Première passe : calcul basé sur les ratios
        for slug in self.sequence.get_sequence():
            temp_stage = self.registry.create_stage(slug, self.device)
            ratio = temp_stage.config.epochs_ratio
            
            raw_epochs = self.total_epochs_planned * ratio
            epochs = int(raw_epochs)
            
            epochs_per_stage[slug] = epochs
            
            print(f"  Stage '{slug}': ratio={ratio:.3f}, "
                  f"calcul={self.total_epochs_planned}*{ratio:.3f}={raw_epochs:.2f}, "
                  f"arrondi={epochs}")
        
        # Deuxième passe : ajustement pour atteindre exactement le total prévu
        total_calculated = sum(epochs_per_stage.values())
        print(f"\nSomme initiale des époques: {total_calculated}/{self.total_epochs_planned}")
        
        if total_calculated != self.total_epochs_planned:
            last_slug = self.sequence.get_sequence()[-1]
            adjustment = self.total_epochs_planned - total_calculated
            
            print(f"  Ajustement requis: {adjustment} époques")
            print(f"  Appliqué au dernier stage ('{last_slug}'): "
                  f"{epochs_per_stage[last_slug]} → {epochs_per_stage[last_slug] + adjustment}")
            
            epochs_per_stage[last_slug] += adjustment
        else:
            print("  Aucun ajustement nécessaire, la somme est exacte")
        
        # Vérification finale
        print("\n=== Répartition finale des époques ===")
        for slug in self.sequence.get_sequence():
            status = "⚠️ AUCUNE ÉPOQUE" if epochs_per_stage[slug] == 0 else "✓"
            print(f"  Stage '{slug}': {epochs_per_stage[slug]} époques {status}")
        
        print(f"Total final: {sum(epochs_per_stage.values())}/{self.total_epochs_planned}\n")
        
        return epochs_per_stage
    
    def initialize_stage(self, slug: str) -> BaseStage:
        """
        Initialise un stage spécifique par son slug.
        
        Args:
            slug: Identifiant unique du stage
            
        Returns:
            Instance du stage initialisé
        """
        print(f"\n🎯 Initialisation du Stage '{slug}'...")
        
        # Paramètres spécialisés selon le stage et ses capacités
        stage_class = self.registry.get_stage(slug)
        stage_kwargs = self._get_stage_kwargs(stage_class)
        
        # Création du stage
        stage = self.registry.create_stage(slug, self.device, **stage_kwargs)
        
        # Configuration des données d'entraînement
        training_config = stage.prepare_training_data(self.global_config)
        stage.training_config = training_config
        
        # Stockage dans les stages actifs
        self.active_stages[slug] = stage
        
        print(f"✅ Stage '{slug}' ({stage.config.description}) initialisé")
        print(f"   ⏱️  Époques prévues: {self.epochs_per_stage[slug]}")
        print(f"   🎯 Seuil convergence: {stage.config.convergence_threshold}")
        
        return stage
    
    def _get_stage_kwargs(self, stage_class) -> Dict[str, Any]:
        """
        Détermine les paramètres à passer au constructeur d'un stage.
        
        Args:
            stage_class: Classe du stage
            
        Returns:
            Dictionnaire des paramètres compatibles
        """
        import inspect
        
        sig = inspect.signature(stage_class.__init__)
        param_names = list(sig.parameters.keys())
        
        stage_kwargs = {}
        
        if 'min_obstacle_size' in param_names and hasattr(self.global_config, 'MIN_OBSTACLE_SIZE'):
            stage_kwargs['min_obstacle_size'] = self.global_config.MIN_OBSTACLE_SIZE
        
        if 'max_obstacle_size' in param_names and hasattr(self.global_config, 'MAX_OBSTACLE_SIZE'):
            stage_kwargs['max_obstacle_size'] = self.global_config.MAX_OBSTACLE_SIZE
        
        return stage_kwargs
    
    def execute_stage(self, slug: str, trainer_callback,
                     early_stopping: bool = True) -> Dict[str, Any]:
        """
        Exécute un stage spécifique par son slug.
        
        Args:
            slug: Identifiant unique du stage
            trainer_callback: Fonction callback pour l'entraînement
            early_stopping: Active l'arrêt précoce
            
        Returns:
            Métriques de l'exécution du stage
            
        Raises:
            ValueError: Si le nombre d'époques alloué au stage est zéro
        """
        if slug not in self.active_stages:
            stage = self.initialize_stage(slug)
        else:
            stage = self.active_stages[slug]
        
        max_epochs = self.epochs_per_stage[slug]
        
        if max_epochs == 0:
            raise ValueError(f"Le Stage '{slug}' ({stage.config.description}) a 0 époque allouée. "
                           f"Chaque stage doit avoir au moins une époque d'entraînement.")
        
        print(f"\n🚀 === EXÉCUTION STAGE '{slug}' - {stage.config.description.upper()} ===")
        print(f"📊 Maximum {max_epochs} époques")
        
        stage.reset_training_history()
        
        execution_result = trainer_callback(stage, max_epochs, early_stopping)
        
        stage_summary = stage.get_stage_summary()
        stage_summary.update(execution_result)
        
        # Sauvegarde automatique du checkpoint si configuré
        if hasattr(self.global_config, 'SAVE_STAGE_CHECKPOINTS') and self.global_config.SAVE_STAGE_CHECKPOINTS:
            self.save_stage_checkpoint_with_callback(slug, trainer_callback)
        
        # Enregistrement dans l'historique
        self.execution_history.append({
            'stage_slug': slug,
            'timestamp': self._get_timestamp(),
            'summary': stage_summary
        })
        
        print(f"✅ === STAGE '{slug}' TERMINÉ ===")
        
        return stage_summary
    
    def save_stage_checkpoint_with_callback(self, slug: str, trainer_callback):
        """Sauvegarde le checkpoint d'un stage avec callback vers le trainer."""
        if hasattr(trainer_callback, '__self__'):
            trainer = trainer_callback.__self__
            trainer.save_stage_checkpoint(slug)
    
    def execute_full_curriculum(self, trainer_callback) -> Dict[str, Any]:
        """
        Exécute le curriculum complet de tous les stages.
        
        Args:
            trainer_callback: Fonction callback pour l'entraînement
            
        Returns:
            Métriques globales de l'exécution
        """
        print(f"\n🚀 === DÉBUT CURRICULUM MODULAIRE ===")
        print(f"🔄 Séquence: {' → '.join(self.sequence.get_sequence())}")
        print(f"📊 Époques totales: {self.total_epochs_planned}")
        
        import time
        start_time = time.time()
        
        all_stage_results = {}
        total_epochs_actual = 0
        
        # Exécution séquentielle des stages
        for slug in self.sequence.get_sequence():
            stage_result = self.execute_stage(slug, trainer_callback)
            all_stage_results[slug] = stage_result
            total_epochs_actual += stage_result.get('epochs_trained', 0)
        
        # Métriques globales
        total_time = time.time() - start_time
        
        global_results = {
            'curriculum_sequence': self.sequence.get_sequence(),
            'total_epochs_planned': self.total_epochs_planned,
            'total_epochs_actual': total_epochs_actual,
            'total_time_seconds': total_time,
            'total_time_formatted': f"{total_time/60:.1f} min",
            'stage_results': all_stage_results,
            'all_stages_converged': all(
                result.get('converged', False)
                for result in all_stage_results.values()
            ),
            'execution_history': self.execution_history
        }
        
        # Résumé final
        final_slug = self.sequence.get_sequence()[-1]
        final_loss = all_stage_results[final_slug].get('final_loss', float('inf'))
        global_results['final_loss'] = final_loss
        
        print(f"\n🎉 === CURRICULUM MODULAIRE TERMINÉ ===")
        print(f"⏱️  Temps total: {total_time/60:.1f} minutes")
        print(f"📊 Époques: {total_epochs_actual}/{self.total_epochs_planned}")
        print(f"🎯 Convergence globale: {'✅' if global_results['all_stages_converged'] else '❌'}")
        print(f"📉 Perte finale: {final_loss:.6f}")
        
        return global_results
    
    def add_custom_stage(self, slug: str, stage_class):
        """
        Ajoute un stage personnalisé au gestionnaire.
        
        Args:
            slug: Identifiant unique pour le nouveau stage
            stage_class: Classe du stage personnalisé
        """
        # Note: Avec l'auto-discovery, il suffit de créer un répertoire
        # Mais on garde cette méthode pour la compatibilité
        print(f"⚠️  add_custom_stage est déprécié. Créez simplement un répertoire stages/{slug}/")
    
    def save_stage_checkpoint(self, slug: str, model_state: Dict[str, Any],
                             output_dir: Path):
        """
        Sauvegarde le checkpoint d'un stage.
        
        Args:
            slug: Identifiant du stage
            model_state: État du modèle à sauvegarder
            output_dir: Répertoire de sortie
        """
        if slug not in self.active_stages:
            print(f"⚠️  Stage '{slug}' non actif, checkpoint ignoré")
            return
        
        stage = self.active_stages[slug]
        stage_dir = output_dir / f"stage_{slug}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarde des données du stage
        checkpoint_data = {
            'stage_slug': slug,
            'stage_config': stage.config.__dict__,
            'training_history': stage.training_history,
            'model_state': model_state,
            'timestamp': self._get_timestamp()
        }
        
        # Sauvegarde spécialisée pour les stages avec statistiques d'intensité
        if hasattr(stage, 'get_intensity_statistics'):
            checkpoint_data['intensity_statistics'] = stage.get_intensity_statistics()
        
        if hasattr(stage, 'get_attenuation_statistics'):
            checkpoint_data['attenuation_statistics'] = stage.get_attenuation_statistics()
        
        # Fichiers de sauvegarde
        checkpoint_path = stage_dir / "stage_checkpoint.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
        
        print(f"💾 Checkpoint Stage '{slug}' sauvegardé: {stage_dir}")
    
    def get_stage(self, slug: str) -> Optional[BaseStage]:
        """Récupère un stage actif par son slug."""
        return self.active_stages.get(slug)
    
    def _get_timestamp(self) -> str:
        """Génère un timestamp pour les logs."""
        import datetime
        return datetime.datetime.now().isoformat()
    
    def get_curriculum_summary(self) -> Dict[str, Any]:
        """Génère un résumé du curriculum configuré."""
        return {
            'stage_sequence': self.sequence.get_sequence(),
            'epochs_per_stage': self.epochs_per_stage,
            'total_epochs': self.total_epochs_planned,
            'available_stages': self.registry.list_stages(),
            'active_stages': list(self.active_stages.keys())
        }
