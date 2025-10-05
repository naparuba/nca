"""
Gestionnaire de stages modulaire pour le NCA.
Orchestre l'exécution des stages de manière découplée et extensible.
"""

from typing import Dict, Any, List, Optional, Type
import torch
import json
from pathlib import Path

from .base_stage import BaseStage
from .stage1 import Stage1
from .stage2 import Stage2
from .stage3 import Stage3
from .stage4 import Stage4


class StageRegistry:
    """
    Registre des stages disponibles.
    Facilite l'ajout de nouveaux stages sans modification du code existant.
    """
    
    def __init__(self):
        self._stages: Dict[int, Type[BaseStage]] = {}
        self._register_default_stages()
    
    def _register_default_stages(self):
        """Enregistre les stages par défaut."""
        self.register_stage(1, Stage1)
        self.register_stage(2, Stage2)
        self.register_stage(3, Stage3)
        self.register_stage(4, Stage4)
    
    def register_stage(self, stage_id: int, stage_class: Type[BaseStage]):
        """
        Enregistre un nouveau stage.
        
        Args:
            stage_id: Identifiant unique du stage
            stage_class: Classe du stage à enregistrer
        """
        if not issubclass(stage_class, BaseStage):
            raise ValueError(f"Stage class must inherit from BaseStage")
        
        self._stages[stage_id] = stage_class
        print(f"✅ Stage {stage_id} ({stage_class.__name__}) enregistré")
    
    def get_stage_class(self, stage_id: int) -> Type[BaseStage]:
        """Récupère la classe d'un stage par son ID."""
        if stage_id not in self._stages:
            raise ValueError(f"Stage {stage_id} non trouvé dans le registre")
        return self._stages[stage_id]
    
    def list_available_stages(self) -> List[int]:
        """Liste les IDs des stages disponibles."""
        return sorted(self._stages.keys())
    
    def create_stage(self, stage_id: int, device: str = "cpu", **kwargs) -> BaseStage:
        """
        Crée une instance d'un stage.
        
        Args:
            stage_id: ID du stage à créer
            device: Device PyTorch
            **kwargs: Arguments supplémentaires pour le constructeur
            
        Returns:
            Instance du stage
        """
        stage_class = self.get_stage_class(stage_id)
        return stage_class(device=device, **kwargs)


class ModularStageManager:
    """
    Gestionnaire principal des stages modulaires.
    Orchestre l'exécution séquentielle des stages de manière découplée.
    """
    
    def __init__(self, global_config: Any, device: str = "cpu"):
        self.global_config = global_config
        self.device = device
        self.registry = StageRegistry()
        self.active_stages: Dict[int, BaseStage] = {}
        self.execution_history: List[Dict[str, Any]] = []
        
        # Configuration d'exécution
        self.stage_sequence = self._determine_stage_sequence()
        self.total_epochs_planned = global_config.TOTAL_EPOCHS
        self.epochs_per_stage = self._calculate_epochs_per_stage()
    
    def _determine_stage_sequence(self) -> List[int]:
        """Détermine la séquence d'exécution des stages."""
        # Par défaut : tous les stages disponibles dans l'ordre
        available_stages = self.registry.list_available_stages()
        
        # Possibilité de personnaliser via la configuration
        if hasattr(self.global_config, 'CUSTOM_STAGE_SEQUENCE'):
            sequence = self.global_config.CUSTOM_STAGE_SEQUENCE
            # Validation
            for stage_id in sequence:
                if stage_id not in available_stages:
                    raise ValueError(f"Stage {stage_id} non disponible")
            return sequence
        
        return available_stages
    
    def _calculate_epochs_per_stage(self) -> Dict[int, int]:
        """Calcule le nombre d'époques par stage selon leur configuration."""
        epochs_per_stage = {}
        
        for stage_id in self.stage_sequence:
            # Créer temporairement le stage pour récupérer sa config
            temp_stage = self.registry.create_stage(stage_id, self.device)
            ratio = temp_stage.config.epochs_ratio
            epochs = int(self.total_epochs_planned * ratio)
            epochs_per_stage[stage_id] = epochs
        
        # Ajustement pour s'assurer que la somme = total prévu
        total_calculated = sum(epochs_per_stage.values())
        if total_calculated != self.total_epochs_planned:
            # Ajuste le dernier stage
            last_stage = self.stage_sequence[-1]
            adjustment = self.total_epochs_planned - total_calculated
            epochs_per_stage[last_stage] += adjustment
        
        return epochs_per_stage
    
    def initialize_stage(self, stage_id: int) -> BaseStage:
        """
        Initialise un stage spécifique.
        
        Args:
            stage_id: ID du stage à initialiser
            
        Returns:
            Instance du stage initialisé
        """
        print(f"\n🎯 Initialisation du Stage {stage_id}...")
        
        # Paramètres spécialisés selon le stage et ses capacités
        stage_class = self.registry.get_stage_class(stage_id)
        stage_kwargs = self._get_stage_kwargs(stage_class)
        
        # Création du stage
        stage = self.registry.create_stage(stage_id, self.device, **stage_kwargs)
        
        # Configuration des données d'entraînement
        training_config = stage.prepare_training_data(self.global_config)
        stage.training_config = training_config
        
        # Stockage dans les stages actifs
        self.active_stages[stage_id] = stage
        
        print(f"✅ Stage {stage_id} ({stage.config.name}) initialisé")
        print(f"   📋 {stage.config.description}")
        print(f"   ⏱️  Époques prévues: {self.epochs_per_stage[stage_id]}")
        print(f"   🎯 Seuil convergence: {stage.config.convergence_threshold}")
        
        return stage
    
    def _get_stage_kwargs(self, stage_class: Type[BaseStage]) -> Dict[str, Any]:
        """
        Détermine les paramètres à passer au constructeur d'un stage.
        
        Args:
            stage_class: Classe du stage
            
        Returns:
            Dictionnaire des paramètres compatibles
        """
        import inspect
        
        # Récupération de la signature du constructeur
        sig = inspect.signature(stage_class.__init__)
        param_names = list(sig.parameters.keys())
        
        stage_kwargs = {}
        
        # Ajout conditionnel des paramètres selon ce que le stage accepte
        if 'min_obstacle_size' in param_names and hasattr(self.global_config, 'MIN_OBSTACLE_SIZE'):
            stage_kwargs['min_obstacle_size'] = self.global_config.MIN_OBSTACLE_SIZE
        
        if 'max_obstacle_size' in param_names and hasattr(self.global_config, 'MAX_OBSTACLE_SIZE'):
            stage_kwargs['max_obstacle_size'] = self.global_config.MAX_OBSTACLE_SIZE
        
        return stage_kwargs
    
    def execute_stage(self, stage_id: int, trainer_callback,
                     early_stopping: bool = True) -> Dict[str, Any]:
        """
        Exécute un stage spécifique.
        
        Args:
            stage_id: ID du stage à exécuter
            trainer_callback: Fonction callback pour l'entraînement
            early_stopping: Active l'arrêt précoce
            
        Returns:
            Métriques de l'exécution du stage
        """
        if stage_id not in self.active_stages:
            stage = self.initialize_stage(stage_id)
        else:
            stage = self.active_stages[stage_id]
        
        max_epochs = self.epochs_per_stage[stage_id]
        
        print(f"\n🚀 === EXÉCUTION STAGE {stage_id} - {stage.config.name.upper()} ===")
        print(f"📊 Maximum {max_epochs} époques")
        
        # Réinitialisation de l'historique du stage
        stage.reset_training_history()
        
        # Exécution de l'entraînement via callback
        execution_result = trainer_callback(stage, max_epochs, early_stopping)
        
        # Génération du résumé
        stage_summary = stage.get_stage_summary()
        stage_summary.update(execution_result)
        
        # NOUVEAU : Sauvegarde automatique du checkpoint si configuré
        if hasattr(self.global_config, 'SAVE_STAGE_CHECKPOINTS') and self.global_config.SAVE_STAGE_CHECKPOINTS:
            self.save_stage_checkpoint_with_callback(stage_id, trainer_callback)
        
        # Enregistrement dans l'historique
        self.execution_history.append({
            'stage_id': stage_id,
            'timestamp': self._get_timestamp(),
            'summary': stage_summary
        })
        
        print(f"✅ === STAGE {stage_id} TERMINÉ ===")
        
        return stage_summary
    
    def save_stage_checkpoint_with_callback(self, stage_id: int, trainer_callback):
        """Sauvegarde le checkpoint d'un stage avec callback vers le trainer."""
        if hasattr(trainer_callback, '__self__'):
            trainer = trainer_callback.__self__
            trainer.save_stage_checkpoint(stage_id)
    
    def execute_full_curriculum(self, trainer_callback) -> Dict[str, Any]:
        """
        Exécute le curriculum complet de tous les stages.
        
        Args:
            trainer_callback: Fonction callback pour l'entraînement
            
        Returns:
            Métriques globales de l'exécution
        """
        print(f"\n🚀 === DÉBUT CURRICULUM MODULAIRE ===")
        print(f"🔄 Séquence: {self.stage_sequence}")
        print(f"📊 Époques totales: {self.total_epochs_planned}")
        
        import time
        start_time = time.time()
        
        all_stage_results = {}
        total_epochs_actual = 0
        
        # Exécution séquentielle des stages
        for stage_id in self.stage_sequence:
            stage_result = self.execute_stage(stage_id, trainer_callback)
            all_stage_results[stage_id] = stage_result
            total_epochs_actual += stage_result.get('epochs_trained', 0)
        
        # Métriques globales
        total_time = time.time() - start_time
        
        global_results = {
            'curriculum_sequence': self.stage_sequence,
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
        final_stage_id = self.stage_sequence[-1]
        final_loss = all_stage_results[final_stage_id].get('final_loss', float('inf'))
        global_results['final_loss'] = final_loss
        
        print(f"\n🎉 === CURRICULUM MODULAIRE TERMINÉ ===")
        print(f"⏱️  Temps total: {total_time/60:.1f} minutes")
        print(f"📊 Époques: {total_epochs_actual}/{self.total_epochs_planned}")
        print(f"🎯 Convergence globale: {'✅' if global_results['all_stages_converged'] else '❌'}")
        print(f"📉 Perte finale: {final_loss:.6f}")
        
        return global_results
    
    def add_custom_stage(self, stage_id: int, stage_class: Type[BaseStage]):
        """
        Ajoute un stage personnalisé au gestionnaire.
        
        Args:
            stage_id: ID unique pour le nouveau stage
            stage_class: Classe du stage personnalisé
        """
        self.registry.register_stage(stage_id, stage_class)
        
        # Recalcul de la séquence et époques si nécessaire
        self.stage_sequence = self._determine_stage_sequence()
        self.epochs_per_stage = self._calculate_epochs_per_stage()
    
    def save_stage_checkpoint(self, stage_id: int, model_state: Dict[str, Any],
                             output_dir: Path):
        """
        Sauvegarde le checkpoint d'un stage.
        
        Args:
            stage_id: ID du stage
            model_state: État du modèle à sauvegarder
            output_dir: Répertoire de sortie
        """
        if stage_id not in self.active_stages:
            print(f"⚠️  Stage {stage_id} non actif, checkpoint ignoré")
            return
        
        stage = self.active_stages[stage_id]
        stage_dir = output_dir / f"stage_{stage_id}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarde des données du stage
        checkpoint_data = {
            'stage_config': stage.config.__dict__,
            'training_history': stage.training_history,
            'model_state': model_state,
            'timestamp': self._get_timestamp()
        }
        
        # Sauvegarde spécialisée pour Stage 4 (intensités)
        if hasattr(stage, 'get_intensity_statistics'):
            checkpoint_data['intensity_statistics'] = stage.get_intensity_statistics()
        
        # Fichiers de sauvegarde
        checkpoint_path = stage_dir / "stage_checkpoint.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
        
        print(f"💾 Checkpoint Stage {stage_id} sauvegardé: {stage_dir}")
    
    def get_stage(self, stage_id: int) -> Optional[BaseStage]:
        """Récupère un stage actif par son ID."""
        return self.active_stages.get(stage_id)
    
    def _get_timestamp(self) -> str:
        """Génère un timestamp pour les logs."""
        import datetime
        return datetime.datetime.now().isoformat()
    
    def get_curriculum_summary(self) -> Dict[str, Any]:
        """Génère un résumé du curriculum configuré."""
        return {
            'stage_sequence': self.stage_sequence,
            'epochs_per_stage': self.epochs_per_stage,
            'total_epochs': self.total_epochs_planned,
            'available_stages': self.registry.list_available_stages(),
            'active_stages': list(self.active_stages.keys())
        }
