"""
Script de test du nouveau système modulaire refactoré v11.
Vérifie que l'auto-découverte et la séquence fonctionnent correctement.
"""

import sys
from pathlib import Path

# Ajout du chemin pour les imports
sys.path.insert(0, str(Path(__file__).parent))

from stages import StageAutoRegistry, StageSequence, ModularStageManager


def test_stage_discovery():
    """Test de l'auto-découverte des stages."""
    print("\n" + "="*60)
    print("TEST 1 : Auto-découverte des stages")
    print("="*60)
    
    registry = StageAutoRegistry()
    stages = registry.list_stages()
    
    print(f"\n✅ {len(stages)} stages découverts :")
    for slug in stages:
        stage_class = registry.get_stage(slug)
        print(f"   - '{slug}' → {stage_class.__name__}")
    
    # Vérification que tous les stages attendus sont présents
    expected = ['multiple_obstacles', 'no_obstacles', 'single_obstacle',
                'time_attenuation', 'variable_intensity']
    
    for slug in expected:
        assert slug in stages, f"Stage '{slug}' non trouvé !"
    
    print("\n✅ Tous les stages attendus sont présents")
    return True


def test_sequence():
    """Test de la séquence d'exécution."""
    print("\n" + "="*60)
    print("TEST 2 : Séquence d'exécution")
    print("="*60)
    
    sequence = StageSequence()
    seq_list = sequence.get_sequence()
    
    print(f"\n✅ Séquence définie :")
    for i, slug in enumerate(seq_list, 1):
        print(f"   {i}. {slug}")
    
    # Vérification de l'ordre
    expected_order = [
        'no_obstacles',
        'single_obstacle',
        'multiple_obstacles',
        'variable_intensity',
        'time_attenuation',
    ]
    
    assert seq_list == expected_order, "L'ordre de la séquence ne correspond pas !"
    
    print("\n✅ L'ordre de la séquence est correct")
    return True


def test_stage_instantiation():
    """Test de l'instanciation des stages."""
    print("\n" + "="*60)
    print("TEST 3 : Instanciation des stages")
    print("="*60)
    
    registry = StageAutoRegistry()
    
    for slug in registry.list_stages():
        try:
            stage = registry.create_stage(slug, device='cpu')
            print(f"   ✅ Stage '{slug}' instancié : {stage.__class__.__name__}")
            
            # Vérification des attributs essentiels
            assert hasattr(stage, 'config'), f"Stage '{slug}' n'a pas de config"
            assert stage.config.name == slug, f"Le slug ne correspond pas : {stage.config.name} != {slug}"
            assert hasattr(stage, 'generate_environment'), f"Stage '{slug}' n'a pas generate_environment"
            assert hasattr(stage, 'prepare_training_data'), f"Stage '{slug}' n'a pas prepare_training_data"
            assert hasattr(stage, 'validate_convergence'), f"Stage '{slug}' n'a pas validate_convergence"
            
        except Exception as e:
            print(f"   ❌ Erreur lors de l'instanciation de '{slug}': {e}")
            return False
    
    print("\n✅ Tous les stages peuvent être instanciés correctement")
    return True


def test_visualizer_discovery():
    """Test de l'auto-découverte des visualiseurs."""
    print("\n" + "="*60)
    print("TEST 4 : Auto-découverte des visualiseurs")
    print("="*60)
    
    from stages.visualizers import get_visualizer
    
    registry = StageAutoRegistry()
    
    for slug in registry.list_stages():
        visualizer = get_visualizer(slug)
        if visualizer:
            print(f"   ✅ Visualiseur '{slug}' trouvé : {visualizer.__class__.__name__}")
        else:
            print(f"   ⚠️  Aucun visualiseur pour '{slug}'")
    
    print("\n✅ Auto-découverte des visualiseurs terminée")
    return True


def test_sequence_manipulation():
    """Test des manipulations de séquence."""
    print("\n" + "="*60)
    print("TEST 5 : Manipulation de la séquence")
    print("="*60)
    
    # Création d'une séquence personnalisée
    custom_seq = ['no_obstacles', 'single_obstacle']
    sequence = StageSequence(custom_seq)
    
    print(f"\n📋 Séquence initiale : {sequence.get_sequence()}")
    
    # Test d'insertion
    sequence.insert_after('multiple_obstacles', 'single_obstacle')
    print(f"   Après insertion : {sequence.get_sequence()}")
    
    # Test d'ajout à la fin
    sequence.append('variable_intensity')
    print(f"   Après append : {sequence.get_sequence()}")
    
    # Test de suppression
    sequence.remove('multiple_obstacles')
    print(f"   Après suppression : {sequence.get_sequence()}")
    
    expected = ['no_obstacles', 'single_obstacle', 'variable_intensity']
    assert sequence.get_sequence() == expected, "La manipulation de séquence a échoué"
    
    print("\n✅ Manipulation de séquence réussie")
    return True


def main():
    """Fonction principale de test."""
    print("\n" + "="*60)
    print("🧪 TESTS DU SYSTÈME MODULAIRE REFACTORÉ v11")
    print("="*60)
    
    tests = [
        test_stage_discovery,
        test_sequence,
        test_stage_instantiation,
        test_visualizer_discovery,
        test_sequence_manipulation,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append((test_func.__name__, result))
        except Exception as e:
            print(f"\n❌ ERREUR dans {test_func.__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_func.__name__, False))
    
    # Résumé
    print("\n" + "="*60)
    print("📊 RÉSUMÉ DES TESTS")
    print("="*60)
    
    for test_name, result in results:
        status = "✅ RÉUSSI" if result else "❌ ÉCHOUÉ"
        print(f"   {status} : {test_name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n🎉 TOUS LES TESTS SONT PASSÉS !")
        print("✅ Le système modulaire refactoré v11 est opérationnel")
    else:
        print("\n⚠️  CERTAINS TESTS ONT ÉCHOUÉ")
        print("❌ Des corrections sont nécessaires")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

