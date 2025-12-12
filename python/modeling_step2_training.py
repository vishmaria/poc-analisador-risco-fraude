# ============================================================================
# fraud_modeling_step2_training.py - Treino de Modelos
# ============================================================================

"""
Treina 4 modelos diferentes com tratamento de desbalanceamento
Usa TRAIN.parquet para treino/validação interna
Avalia em TEST.parquet
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, 
    f1_score, recall_score, precision_score, confusion_matrix,
    classification_report, auc
)
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

CONFIG = {
    'dtypes': {
        'amount': 'float32',
        'newBalanceDest': 'float32',
        'isFraud': 'uint8',
        'CASH_IN': 'uint8', 'CASH_OUT': 'uint8', 'DEBIT': 'uint8',
        'PAYMENT': 'uint8', 'TRANSFER': 'uint8',
        'nameOrig': 'uint32', 'nameDest': 'uint32'
    },
    'feature_cols': [
        'amount', 'newBalanceDest', 
        'CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER',
        'amount_high', 'balance_suspicious', 'num_actions',
        'amount_balance_ratio', 'log_amount'
    ],
    'target_col': 'isFraud',
    'random_state': 42,
    'test_size': 0.2
}

# ============================================================================
# TRATAMENTO DE DESBALANCEAMENTO
# ============================================================================

class ImbalancedDataHandler:
    """
    Estratégias para dados desbalanceados
    Dataset com 9% fraude = 11:1 ratio
    """
    
    @staticmethod
    def apply_class_weights(y_train):
        """Calcula pesos das classes para modelos"""
        fraud_count = (y_train == 1).sum()
        legit_count = (y_train == 0).sum()
        
        weight_fraud = legit_count / fraud_count
        weight_legit = 1.0
        
        class_weights = {
            0: weight_legit,
            1: weight_fraud
        }
        
        print(f"\n⚖️  PESOS DAS CLASSES:")
        print(f"   Legítima: {weight_legit:.2f}")
        print(f"   Fraude:   {weight_fraud:.2f}")
        
        return class_weights
    
    @staticmethod
    def apply_smote(X_train, y_train):
        """
        SMOTE: Synthetic Minority Oversampling Technique
        Cria fraudes sintéticas para balancear dataset
        """
        try:
            from imblearn.over_sampling import SMOTE
            
            print(f"\n🔄 Aplicando SMOTE...")
            smote = SMOTE(random_state=CONFIG['random_state'], k_neighbors=5)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            
            print(f"   Antes: {len(X_train)} amostras")
            print(f"   Depois: {len(X_train_balanced)} amostras")
            print(f"   Nova proporção: {(y_train_balanced == 1).sum() / len(y_train_balanced) * 100:.1f}% fraude")
            
            return X_train_balanced, y_train_balanced
        
        except ImportError:
            print(f"⚠️  SMOTE não disponível (instale: pip install imbalanced-learn)")
            return X_train, y_train

# ============================================================================
# PREPARAÇÃO DE DADOS
# ============================================================================

class DataPreparation:
    """Prepara dados para modelagem"""
    
    def __init__(self):
        self.scaler = RobustScaler()
        self.feature_cols = CONFIG['feature_cols']
    
    def load_and_prepare(self, train_path, test_path=None):
        """Carrega e prepara datasets"""
        
        print("\n📊 PREPARAÇÃO DE DADOS")
        print("=" * 60)
        
        # Carrega TRAIN
        print(f"\n📂 Carregando TRAIN: {train_path}")
        df_train = pd.read_parquet(train_path)
        
        # Se não tem features engenheiradas, cria aqui
        if 'amount_high' not in df_train.columns:
            print("🔧 Criando features engenheiradas...")
            df_train = self._create_features(df_train)
        
        # Separa X e y
        X_train = df_train[self.feature_cols]
        y_train = df_train[CONFIG['target_col']]
        
        print(f"✓ TRAIN: X={X_train.shape}, y={y_train.shape}")
        
        # Carrega TEST (opcional)
        X_test = None
        y_test = None
        if test_path:
            print(f"\n📂 Carregando TEST: {test_path}")
            df_test = pd.read_parquet(test_path)
            
            if 'amount_high' not in df_test.columns:
                df_test = self._create_features(df_test)
            
            X_test = df_test[self.feature_cols]
            y_test = df_test[CONFIG['target_col']]
            
            print(f"✓ TEST: X={X_test.shape}, y={y_test.shape}")
        
        # Escala features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.feature_cols)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.feature_cols)
        else:
            X_test_scaled = None
        
        print(f"\n✓ Features escaladas com RobustScaler")
        
        return X_train_scaled, y_train, X_test_scaled, y_test
    
    def _create_features(self, df):
        """Cria features engenheiradas"""
        df_feat = df.copy()
        
        amount_mean = df['amount'].mean()
        df_feat['amount_high'] = (df['amount'] > amount_mean).astype(np.uint8)
        
        balance_q25 = df['newBalanceDest'].quantile(0.25)
        balance_q75 = df['newBalanceDest'].quantile(0.75)
        df_feat['balance_suspicious'] = (
            (df['newBalanceDest'] < balance_q25) | 
            (df['newBalanceDest'] > balance_q75)
        ).astype(np.uint8)
        
        action_cols = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
        df_feat['num_actions'] = df[action_cols].sum(axis=1)
        
        df_feat['amount_balance_ratio'] = np.where(
            df['newBalanceDest'] > 0,
            df['amount'] / (df['newBalanceDest'] + 1),
            0
        ).astype(np.float32)
        
        df_feat['log_amount'] = np.log1p(df['amount']).astype(np.float32)
        
        return df_feat

# ============================================================================
# MODELOS
# ============================================================================

class FraudModelTrainer:
    """Treina múltiplos modelos"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.handler = ImbalancedDataHandler()
    
    def train_logistic_regression(self, X_train, y_train, X_test=None, y_test=None):
        """Treina Logistic Regression com class weights"""
        
        print("\n\n🤖 MODELO 1: LOGISTIC REGRESSION")
        print("=" * 60)
        
        class_weights = self.handler.apply_class_weights(y_train)
        
        model = LogisticRegression(
            class_weight=class_weights,
            max_iter=1000,
            random_state=CONFIG['random_state'],
            solver='lbfgs',
            n_jobs=-1
        )
        
        print(f"🔄 Treinando Logistic Regression...")
        model.fit(X_train, y_train)
        
        # Avalia
        results = self._evaluate_model(
            model, X_train, y_train, X_test, y_test, 
            model_name='Logistic Regression'
        )
        
        self.models['logistic_regression'] = model
        self.results['logistic_regression'] = results
        
        return model, results
    
    def train_random_forest(self, X_train, y_train, X_test=None, y_test=None):
        """Treina Random Forest com class weights"""
        
        print("\n\n🤖 MODELO 2: RANDOM FOREST")
        print("=" * 60)
        
        class_weights = self.handler.apply_class_weights(y_train)
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            class_weight=class_weights,
            n_jobs=-1,
            random_state=CONFIG['random_state']
        )
        
        print(f"🔄 Treinando Random Forest (100 árvores)...")
        model.fit(X_train, y_train)
        
        results = self._evaluate_model(
            model, X_train, y_train, X_test, y_test, 
            model_name='Random Forest'
        )
        
        self.models['random_forest'] = model
        self.results['random_forest'] = results
        
        return model, results
    
    def train_xgboost(self, X_train, y_train, X_test=None, y_test=None):
        """Treina XGBoost com scale_pos_weight"""
        
        print("\n\n🤖 MODELO 3: XGBOOST")
        print("=" * 60)
        
        # Calcula weight para fraude
        fraud_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        print(f"⚖️  Scale pos weight: {fraud_weight:.2f}")
        
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=fraud_weight,
            random_state=CONFIG['random_state'],
            n_jobs=-1,
            eval_metric='auc'
        )
        
        print(f"🔄 Treinando XGBoost...")
        model.fit(X_train, y_train)
        
        results = self._evaluate_model(
            model, X_train, y_train, X_test, y_test, 
            model_name='XGBoost'
        )
        
        self.models['xgboost'] = model
        self.results['xgboost'] = results
        
        return model, results
    
    def train_lightgbm(self, X_train, y_train, X_test=None, y_test=None):
        """Treina LightGBM com scale_pos_weight"""
        
        print("\n\n🤖 MODELO 4: LIGHTGBM")
        print("=" * 60)
        
        fraud_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        print(f"⚖️  Scale pos weight: {fraud_weight:.2f}")
        
        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=fraud_weight,
            random_state=CONFIG['random_state'],
            n_jobs=-1
        )
        
        print(f"🔄 Treinando LightGBM...")
        model.fit(X_train, y_train)
        
        results = self._evaluate_model(
            model, X_train, y_train, X_test, y_test, 
            model_name='LightGBM'
        )
        
        self.models['lightgbm'] = model
        self.results['lightgbm'] = results
        
        return model, results
    
    def _evaluate_model(self, model, X_train, y_train, X_test=None, y_test=None, model_name=''):
        """Avalia modelo em TRAIN e TEST"""
        
        results = {}
        
        # Previsão em TRAIN
        y_pred_train = model.predict(X_train)
        y_pred_proba_train = model.predict_proba(X_train)[:, 1]
        
        roc_auc_train = roc_auc_score(y_train, y_pred_proba_train)
        f1_train = f1_score(y_train, y_pred_train)
        recall_train = recall_score(y_train, y_pred_train)
        precision_train = precision_score(y_train, y_pred_train)
        
        print(f"\n📊 PERFORMANCE NO TRAIN:")
        print(f"   ROC-AUC:  {roc_auc_train:.4f}")
        print(f"   F1-Score: {f1_train:.4f}")
        print(f"   Recall:   {recall_train:.4f}")
        print(f"   Precision:{precision_train:.4f}")
        
        results['train'] = {
            'roc_auc': roc_auc_train,
            'f1': f1_train,
            'recall': recall_train,
            'precision': precision_train
        }
        
        # Previsão em TEST (se disponível)
        if X_test is not None and y_test is not None:
            y_pred_test = model.predict(X_test)
            y_pred_proba_test = model.predict_proba(X_test)[:, 1]
            
            roc_auc_test = roc_auc_score(y_test, y_pred_proba_test)
            f1_test = f1_score(y_test, y_pred_test)
            recall_test = recall_score(y_test, y_pred_test)
            precision_test = precision_score(y_test, y_pred_test)
            
            print(f"\n📊 PERFORMANCE NO TEST:")
            print(f"   ROC-AUC:  {roc_auc_test:.4f}")
            print(f"   F1-Score: {f1_test:.4f}")
            print(f"   Recall:   {recall_test:.4f}")
            print(f"   Precision:{precision_test:.4f}")
            
            results['test'] = {
                'roc_auc': roc_auc_test,
                'f1': f1_test,
                'recall': recall_test,
                'precision': precision_test
            }
        
        return results
    
    def compare_models(self):
        """Compara todos os modelos"""
        print("\n\n" + "=" * 60)
        print("🏆 COMPARAÇÃO DE MODELOS")
        print("=" * 60)
        
        comparison = []
        for model_name, metrics in self.results.items():
            if 'test' in metrics:
                roc_auc = metrics['test']['roc_auc']
                f1 = metrics['test']['f1']
                recall = metrics['test']['recall']
                
                comparison.append({
                    'Model': model_name,
                    'ROC-AUC': roc_auc,
                    'F1': f1,
                    'Recall': recall
                })
        
        comparison_df = pd.DataFrame(comparison).sort_values('ROC-AUC', ascending=False)
        
        print("\n")
        print(comparison_df.to_string(index=False))
        
        best_model = comparison_df.iloc[0]['Model']
        best_roc_auc = comparison_df.iloc[0]['ROC-AUC']
        
        print(f"\n🏆 MELHOR MODELO: {best_model.upper()}")
        print(f"   ROC-AUC: {best_roc_auc:.4f}")
        
        return best_model, comparison_df
    
    def save_best_model(self, model_name, path='models/best_fraud_model.pkl'):
        """Salva melhor modelo"""
        model = self.models[model_name]
        
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"\n💾 Modelo salvo em: {path}")

# ============================================================================
# EXECUÇÃO PASSO 2
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "=" * 70)
    print("🚀 PASSO 2: TREINO DE MODELOS")
    print("=" * 70)
    
    # Prepara dados
    prep = DataPreparation()
    X_train, y_train, X_test, y_test = prep.load_and_prepare(
        'datasets/train_engineered.parquet',
        'datasets/test-00000-of-00001.parquet'
    )
    
    # Treina modelos
    trainer = FraudModelTrainer()
    
    trainer.train_logistic_regression(X_train, y_train, X_test, y_test)
    trainer.train_random_forest(X_train, y_train, X_test, y_test)
    trainer.train_xgboost(X_train, y_train, X_test, y_test)
    trainer.train_lightgbm(X_train, y_train, X_test, y_test)
    
    # Compara e seleciona melhor
    best_model, comparison = trainer.compare_models()
    
    # Salva melhor modelo
    import os
    os.makedirs('models', exist_ok=True)
    trainer.save_best_model(best_model)
    
    print("\n" + "=" * 70)
    print("✅ PASSO 2 CONCLUÍDO - MODELOS TREINADOS E AVALIADOS")
    print("=" * 70)
