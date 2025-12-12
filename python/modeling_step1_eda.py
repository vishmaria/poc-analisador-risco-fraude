# 🤖 MODELAGEM DE FRAUDE - GUIA PRÁTICO COMPLETO

## 📊 FASE 2: MODELAGEM (Treino → Teste → Validação)

---

## ARQUITETURA DE MODELAGEM

```
TRAIN.parquet (70%)
    ↓
[1] EDA + Feature Engineering
    ↓
[2] Split: 80% treino / 20% teste interno
    ↓
[3] Treinar Modelos (4 algoritmos)
    ├─ Logistic Regression
    ├─ Random Forest
    ├─ XGBoost
    └─ LightGBM
    ↓
[4] Avaliar com TEST.parquet (15%)
    ↓
[5] Selecionar melhor modelo
    ↓
[6] Validação final com VALIDATION.parquet (15%)
```

---

## PASSO 1: EXPLORAÇÃO E FEATURE ENGINEERING

```python
# ============================================================================
# fraud_modeling_step1_eda.py - Análise e Engenharia de Features
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

class FraudEDA:
    """Exploração de dados e engenharia de features"""
    
    def __init__(self, config_dtypes):
        self.config_dtypes = config_dtypes
        self.scaler = RobustScaler()
        self.feature_stats = {}
    
    def load_data(self, path):
        """Carrega dataset otimizado"""
        print(f"📂 Carregando: {path}")
        df = pd.read_parquet(path, dtype=self.config_dtypes)
        print(f"✓ Shape: {df.shape}")
        print(f"✓ RAM: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        return df
    
    def analyze_target(self, df):
        """Analisa distribuição do target"""
        print("\n📊 DISTRIBUIÇÃO DO TARGET (isFraud)")
        print("=" * 60)
        
        fraud_count = (df['isFraud'] == 1).sum()
        legit_count = (df['isFraud'] == 0).sum()
        fraud_rate = fraud_count / len(df) * 100
        
        print(f"Fraudes:   {fraud_count:,} ({fraud_rate:.2f}%)")
        print(f"Legítimas: {legit_count:,} ({100-fraud_rate:.2f}%)")
        print(f"Desbalanceamento: 1:{legit_count//fraud_count}")
        
        return {
            'fraud_count': fraud_count,
            'legit_count': legit_count,
            'fraud_rate': fraud_rate,
            'imbalance_ratio': legit_count / fraud_count
        }
    
    def analyze_features(self, df):
        """Analisa cada feature"""
        print("\n📈 ANÁLISE DE FEATURES")
        print("=" * 60)
        
        # Amount
        print("\n💵 AMOUNT (valor da transação)")
        fraud_amount = df[df['isFraud'] == 1]['amount']
        legit_amount = df[df['isFraud'] == 0]['amount']
        
        print(f"  Fraude    → Média: ${fraud_amount.mean():,.2f} | Mediana: ${fraud_amount.median():,.2f}")
        print(f"  Legítima  → Média: ${legit_amount.mean():,.2f} | Mediana: ${legit_amount.median():,.2f}")
        print(f"  Diferença: {(fraud_amount.mean() - legit_amount.mean()) / legit_amount.mean() * 100:.1f}%")
        
        # New Balance Dest
        print(f"\n💰 NEW_BALANCE_DEST (saldo destino)")
        fraud_balance = df[df['isFraud'] == 1]['newBalanceDest']
        legit_balance = df[df['isFraud'] == 0]['newBalanceDest']
        
        print(f"  Fraude    → Média: ${fraud_balance.mean():,.2f}")
        print(f"  Legítima  → Média: ${legit_balance.mean():,.2f}")
        
        # Ações
        print(f"\n🎯 FREQUÊNCIA POR AÇÃO")
        actions = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
        
        for action in actions:
            action_df = df[df[action] == 1]
            fraud_in_action = (action_df['isFraud'] == 1).sum()
            fraud_rate = fraud_in_action / len(action_df) * 100 if len(action_df) > 0 else 0
            
            print(f"  {action:10}: {fraud_rate:6.2f}% ({fraud_in_action:,}/{len(action_df):,})")
    
    def engineer_features(self, df):
        """Cria novas features para melhorar modelo"""
        print("\n🔧 ENGENHARIA DE FEATURES")
        print("=" * 60)
        
        df_features = df.copy()
        
        # Feature 1: Amount > média (transação acima do normal)
        amount_mean = df['amount'].mean()
        df_features['amount_high'] = (df['amount'] > amount_mean).astype(np.uint8)
        print(f"✓ amount_high: transações > ${amount_mean:,.2f}")
        
        # Feature 2: Saldo destino suspeito (muito alto ou muito baixo)
        balance_q25 = df['newBalanceDest'].quantile(0.25)
        balance_q75 = df['newBalanceDest'].quantile(0.75)
        df_features['balance_suspicious'] = (
            (df['newBalanceDest'] < balance_q25) | 
            (df['newBalanceDest'] > balance_q75)
        ).astype(np.uint8)
        print(f"✓ balance_suspicious: saldo fora do intervalo IQR")
        
        # Feature 3: Múltiplas ações por registro (não faz sentido, pode ser fraude)
        action_cols = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
        df_features['num_actions'] = df[action_cols].sum(axis=1)
        print(f"✓ num_actions: quantidade de tipos de ação")
        
        # Feature 4: Razão Amount/Balance (eficiência da transação)
        df_features['amount_balance_ratio'] = np.where(
            df['newBalanceDest'] > 0,
            df['amount'] / (df['newBalanceDest'] + 1),
            0
        ).astype(np.float32)
        print(f"✓ amount_balance_ratio: relação entre valor e saldo")
        
        # Feature 5: Log do amount (distribuição mais normal)
        df_features['log_amount'] = np.log1p(df['amount']).astype(np.float32)
        print(f"✓ log_amount: transformação logarítmica do amount")
        
        return df_features
    
    def save_analysis(self, stats, path='eda_report.txt'):
        """Salva relatório da análise"""
        with open(path, 'w') as f:
            f.write("RELATÓRIO DE EDA\n")
            f.write("=" * 60 + "\n")
            f.write(str(stats))
        print(f"\n💾 Relatório salvo em: {path}")

# ============================================================================
# EXECUÇÃO PASSO 1
# ============================================================================

if __name__ == "__main__":
    config_dtypes = {
        'amount': 'float32',
        'newBalanceDest': 'float32',
        'isFraud': 'uint8',
        'CASH_IN': 'uint8', 'CASH_OUT': 'uint8', 'DEBIT': 'uint8',
        'PAYMENT': 'uint8', 'TRANSFER': 'uint8',
        'nameOrig': 'uint32', 'nameDest': 'uint32'
    }
    
    eda = FraudEDA(config_dtypes)
    
    # Carregar dados
    df_train = eda.load_data('datasets/train-00000-of-00001.parquet')
    
    # Análises
    target_stats = eda.analyze_target(df_train)
    eda.analyze_features(df_train)
    
    # Engenharia de features
    df_engineered = eda.engineer_features(df_train)
    
    # Salvar dados processados
    df_engineered.to_parquet('datasets/train_engineered.parquet', compression='snappy')
    print(f"\n✓ Dataset com features salvo em: datasets/train_engineered.parquet")
