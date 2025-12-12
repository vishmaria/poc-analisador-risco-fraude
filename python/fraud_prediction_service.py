# fraud_prediction_service.py
import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.preprocessing import RobustScaler

app = Flask(__name__)

class FraudPredictor:
    def __init__(self, model_path='models/best_fraud_model.pkl'):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        self.scaler = RobustScaler()
        self.feature_stats = self._load_feature_stats()
    
    def _load_feature_stats(self):
        """Carrega estatísticas do treino para feature engineering"""
        df_train = pd.read_parquet('datasets/train_engineered.parquet')
        return {
            'amount_mean': df_train['amount'].mean(),
            'balance_q25': df_train['newBalanceDest'].quantile(0.25),
            'balance_q75': df_train['newBalanceDest'].quantile(0.75)
        }
    
    def preprocess_transaction(self, transaction):
        """Transforma transação em features"""
        # Features base
        features = {
            'amount': transaction['amount'],
            'newBalanceDest': transaction.get('newBalanceDest', 0),
            'CASH_IN': 1 if transaction['type'] == 'CASH_IN' else 0,
            'CASH_OUT': 1 if transaction['type'] == 'CASH_OUT' else 0,
            'DEBIT': 1 if transaction['type'] == 'DEBIT' else 0,
            'PAYMENT': 1 if transaction['type'] == 'PAYMENT' else 0,
            'TRANSFER': 1 if transaction['type'] == 'TRANSFER' else 0
        }
        
        # Features engenheiradas
        features['amount_high'] = 1 if features['amount'] > self.feature_stats['amount_mean'] else 0
        features['balance_suspicious'] = 1 if (
            features['newBalanceDest'] < self.feature_stats['balance_q25'] or
            features['newBalanceDest'] > self.feature_stats['balance_q75']
        ) else 0
        features['num_actions'] = sum([features['CASH_IN'], features['CASH_OUT'], 
                                       features['DEBIT'], features['PAYMENT'], features['TRANSFER']])
        features['amount_balance_ratio'] = features['amount'] / (features['newBalanceDest'] + 1) if features['newBalanceDest'] > 0 else 0
        features['log_amount'] = np.log1p(features['amount'])
        
        return pd.DataFrame([features])
    
    def predict(self, transaction):
        """Predição com probabilidade e threshold"""
        X = self.preprocess_transaction(transaction)
        X_scaled = self.scaler.fit_transform(X)
        
        fraud_probability = self.model.predict_proba(X_scaled)[0][1]
        threshold = 0.423  # Threshold ótimo do step 3
        
        is_fraud = fraud_probability >= threshold
        
        return {
            'fraud_probability': float(fraud_probability),
            'is_fraud': bool(is_fraud),
            'risk_level': 'high' if fraud_probability >= 0.7 else 'medium' if fraud_probability >= 0.4 else 'low',
            'threshold_used': threshold,
            'confidence': float(fraud_probability) if is_fraud else float(1 - fraud_probability)
        }

predictor = FraudPredictor()

@app.route('/predict', methods=['POST'])
def predict_fraud():
    """Endpoint para predição de fraude"""
    try:
        transaction = request.json
        
        # Validação básica
        required_fields = ['amount', 'type']
        if not all(field in transaction for field in required_fields):
            return jsonify({'error': 'Campos obrigatórios: amount, type'}), 400
        
        result = predictor.predict(transaction)
        result['transaction'] = transaction
        result['timestamp'] = pd.Timestamp.now().isoformat()
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
# Adicionar no FINAL do fraud_prediction_service.py:
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'fraud-predictor'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
