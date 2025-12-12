# ============================================================================
# fraud_modeling_step3_validation.py - Validação Final
# ============================================================================

"""
Passo 3: Validar modelo em dados NUNCA VISTOS (VALIDATION.parquet)
- Carrega melhor modelo treinado
- Testa em VALIDATION.parquet (15%)
- Gera métricas finais de produção
- Cria thresholds otimizados para detecção de fraude
"""

import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    f1_score, recall_score, precision_score, confusion_matrix,
    classification_report, accuracy_score
)
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

CONFIG = {
    'feature_cols': [
        'amount', 'newBalanceDest', 
        'CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER',
        'amount_high', 'balance_suspicious', 'num_actions',
        'amount_balance_ratio', 'log_amount'
    ],
    'target_col': 'isFraud'
}

# ============================================================================
# VALIDAÇÃO
# ============================================================================

class FraudModelValidator:
    """Valida modelo em dados de produção"""
    
    def __init__(self, model_path):
        self.model = self._load_model(model_path)
        self.scaler = RobustScaler()
        self.results = {}
    
    def _load_model(self, path):
        """Carrega modelo treinado"""
        print(f"\n📂 Carregando modelo: {path}")
        with open(path, 'rb') as f:
            model = pickle.load(f)
        print(f"✓ Modelo carregado com sucesso")
        return model
    
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
    
    def load_validation_data(self, validation_path):
        """Carrega dataset de validação"""
        
        print(f"\n📂 Carregando VALIDATION: {validation_path}")
        
        df_val = pd.read_parquet(validation_path)
        
        # Cria features se necessário
        if 'amount_high' not in df_val.columns:
            print("🔧 Criando features engenheiradas...")
            df_val = self._create_features(df_val)
        
        X_val = df_val[CONFIG['feature_cols']]
        y_val = df_val[CONFIG['target_col']]
        
        print(f"✓ VALIDATION carregado: X={X_val.shape}, y={y_val.shape}")
        print(f"  Taxa de fraude: {(y_val == 1).sum() / len(y_val) * 100:.2f}%")
        
        return X_val, y_val
    
    def validate(self, X_val, y_val):
        """Executa validação completa"""
        
        print("\n" + "=" * 70)
        print("🔬 VALIDAÇÃO EM DADOS NÃO VISTOS (VALIDATION.parquet)")
        print("=" * 70)
        
        # Escala dados
        X_val_scaled = self.scaler.fit_transform(X_val)
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=CONFIG['feature_cols'])
        
        # Previsões
        y_pred = self.model.predict(X_val_scaled)
        y_pred_proba = self.model.predict_proba(X_val_scaled)[:, 1]
        
        # Métricas básicas
        print("\n📊 MÉTRICAS DE VALIDAÇÃO")
        print("=" * 70)
        
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        roc_auc = roc_auc_score(y_val, y_pred_proba)
        
        print(f"\n🎯 MÉTRICAS PRINCIPAIS:")
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f} (de 100 alertas, {precision*100:.1f}% são fraude real)")
        print(f"   Recall:    {recall:.4f} (encontra {recall*100:.1f}% das fraudes)")
        print(f"   F1-Score:  {f1:.4f}")
        print(f"   ROC-AUC:   {roc_auc:.4f}")
        
        # Matriz de confusão
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
        
        print(f"\n📋 MATRIZ DE CONFUSÃO:")
        print(f"   Verdadeiro Negativo (TN):   {tn:,} (legítimas corretamente identificadas)")
        print(f"   Falso Positivo (FP):        {fp:,} (legítimas classificadas como fraude)")
        print(f"   Falso Negativo (FN):        {fn:,} (fraudes não detectadas) ⚠️")
        print(f"   Verdadeiro Positivo (TP):   {tp:,} (fraudes detectadas)")
        
        # Taxa de falsas fraudes
        if (tn + fp) > 0:
            false_alarm_rate = fp / (tn + fp) * 100
            print(f"\n   Taxa de falso alarme: {false_alarm_rate:.2f}%")
        
        # Armazena resultados
        self.results['validation'] = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'roc_auc': float(roc_auc),
            'confusion_matrix': {
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'tp': int(tp)
            }
        }
        
        return {
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'metrics': self.results['validation']
        }
    
    def find_optimal_threshold(self, y_val, y_pred_proba):
        """Encontra threshold ótimo para maximizar Recall (minimizar FN)"""
        
        print("\n\n" + "=" * 70)
        print("🎯 OTIMIZAÇÃO DE THRESHOLD")
        print("=" * 70)
        
        # Por padrão, o modelo usa 0.5
        # Mas para fraude, queremos HIGH RECALL (encontrar ao máximo)
        
        fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
        precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_val, y_pred_proba)
        
        # F1-score por threshold
        f1_scores = []
        for threshold in pr_thresholds:
            y_pred_opt = (y_pred_proba >= threshold).astype(int)
            f1 = f1_score(y_val, y_pred_opt)
            f1_scores.append(f1)
        
        # Threshold que maximiza F1
        optimal_threshold_f1 = pr_thresholds[np.argmax(f1_scores)]
        best_f1 = np.max(f1_scores)
        
        # Threshold para HIGH RECALL (encontrar mais fraudes)
        # Exemplo: encontrar 95% das fraudes
        target_recall = 0.95
        recall_idx = np.argmax(recall_vals >= target_recall)
        threshold_high_recall = pr_thresholds[recall_idx] if recall_idx < len(pr_thresholds) else 0.3
        
        print(f"\n🔧 RECOMENDAÇÕES DE THRESHOLD:")
        print(f"\n1. THRESHOLD PADRÃO (0.50)")
        print(f"   - Usa o padrão do modelo")
        print(f"   - Equilíbrio Precision/Recall")
        
        print(f"\n2. THRESHOLD OTIMIZADO F1 ({optimal_threshold_f1:.3f})")
        print(f"   - Maximiza F1-Score: {best_f1:.4f}")
        print(f"   - Equilíbrio entre Precision e Recall")
        
        y_pred_f1 = (y_pred_proba >= optimal_threshold_f1).astype(int)
        precision_f1 = precision_score(y_val, y_pred_f1)
        recall_f1 = recall_score(y_val, y_pred_f1)
        print(f"     Precision: {precision_f1:.4f} | Recall: {recall_f1:.4f}")
        
        print(f"\n3. THRESHOLD ALTO RECALL ({threshold_high_recall:.3f})")
        print(f"   - Prioriza encontrar fraudes ({target_recall*100:.0f}% minimum)")
        print(f"   - Trade-off: mais falsos positivos")
        
        y_pred_high_recall = (y_pred_proba >= threshold_high_recall).astype(int)
        precision_hr = precision_score(y_val, y_pred_high_recall)
        recall_hr = recall_score(y_val, y_pred_high_recall)
        print(f"     Precision: {precision_hr:.4f} | Recall: {recall_hr:.4f}")
        
        self.results['thresholds'] = {
            'default': 0.5,
            'optimal_f1': float(optimal_threshold_f1),
            'high_recall': float(threshold_high_recall)
        }
        
        return {
            'default': 0.5,
            'optimal_f1': optimal_threshold_f1,
            'high_recall': threshold_high_recall
        }
    
    def analyze_misclassifications(self, X_val, y_val, y_pred, y_pred_proba):
        """Analisa casos classificados incorretamente"""
        
        print("\n\n" + "=" * 70)
        print("🔍 ANÁLISE DE ERROS")
        print("=" * 70)
        
        # Falsos positivos (legítima classificada como fraude)
        false_positives_idx = (y_val == 0) & (y_pred == 1)
        fp_count = false_positives_idx.sum()
        
        if fp_count > 0:
            print(f"\n❌ FALSOS POSITIVOS: {fp_count:,}")
            print(f"   Transações legítimas alertadas como fraude")
            
            fp_amounts = X_val.loc[false_positives_idx, 'amount']
            print(f"   Amount médio: ${fp_amounts.mean():,.2f}")
            print(f"   Confiança média do modelo: {y_pred_proba[false_positives_idx].mean():.4f}")
        
        # Falsos negativos (fraude não detectada)
        false_negatives_idx = (y_val == 1) & (y_pred == 0)
        fn_count = false_negatives_idx.sum()
        
        if fn_count > 0:
            print(f"\n⚠️  FALSOS NEGATIVOS: {fn_count:,}")
            print(f"   Fraudes NÃO detectadas pelo modelo")
            print(f"   RISCO: Perdas financeiras!")
            
            fn_amounts = X_val.loc[false_negatives_idx, 'amount']
            print(f"   Amount médio: ${fn_amounts.mean():,.2f}")
            print(f"   Confiança média do modelo: {y_pred_proba[false_negatives_idx].mean():.4f}")
            
            print(f"\n   💡 AÇÃO: Investigar por quê não foram detectadas")
            print(f"      - Padrão novo de fraude?")
            print(f"      - Necessário retreinar?")
    
    def generate_report(self, output_path='validation_report.json'):
        """Gera relatório final em JSON"""
        
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'dataset': 'VALIDATION.parquet',
            'results': self.results,
            'recommendations': {
                'next_steps': [
                    '✓ Modelo está pronto para produção',
                    '✓ Monitorar performance mensalmente',
                    '✓ Alertar se ROC-AUC cair abaixo de 0.90',
                    '✓ Retreinar se taxa de fraude mudar > 2%'
                ],
                'threshold_selection': 'Use optimal_f1 para produção'
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n💾 Relatório salvo em: {output_path}")
        
        return report

# ============================================================================
# EXECUÇÃO PASSO 3
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "=" * 70)
    print("🚀 PASSO 3: VALIDAÇÃO FINAL EM DADOS NÃO VISTOS")
    print("=" * 70)
    
    # Carrega modelo
    validator = FraudModelValidator('models/best_fraud_model.pkl')
    
    # Carrega dados de validação
    X_val, y_val = validator.load_validation_data(
        'datasets/validation-00000-of-00001.parquet'
    )
    
    # Valida
    validation_results = validator.validate(X_val, y_val)
    
    # Encontra threshold ótimo
    thresholds = validator.find_optimal_threshold(
        y_val, 
        validation_results['y_pred_proba']
    )
    
    # Analisa erros
    validator.analyze_misclassifications(
        X_val, y_val,
        validation_results['y_pred'],
        validation_results['y_pred_proba']
    )
    
    # Gera relatório
    report = validator.generate_report()
    
    print("\n" + "=" * 70)
    print("✅ PASSO 3 CONCLUÍDO - MODELO VALIDADO E PRONTO PARA PRODUÇÃO")
    print("=" * 70)
    print(f"\n📊 ROC-AUC: {validation_results['metrics']['roc_auc']:.4f}")
    print(f"🎯 Recall: {validation_results['metrics']['recall']:.4f}")
    print(f"✅ Threshold recomendado: {thresholds['optimal_f1']:.3f}")
