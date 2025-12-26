import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from django.core.files.base import ContentFile
import os
from django.db.models import QuerySet
from .models import Medicine
import re
from datetime import date

try:
    # Optional heavy deps for NLP
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    TfidfVectorizer = None
    cosine_similarity = None

try:
    # Optional semantic embeddings for better accuracy
    from sentence_transformers import SentenceTransformer
    _st_model = None
except Exception:
    SentenceTransformer = None
    _st_model = None

class MLPredictor:
    def __init__(self):
        self.models = {}
        self.is_trained = False
        self.training_data = None
        
    def generate_training_data(self):
        """Generate synthetic training data for medicine prices and quantities"""
        np.random.seed(42)
        
        # Generate synthetic medicine data
        n_samples = 1000
        
        # Medicine names (common categories)
        medicine_categories = ['Antibiotic', 'Painkiller', 'Vitamin', 'Antihistamine', 'Antacid']
        medicine_names = []
        for category in medicine_categories:
            for i in range(n_samples // len(medicine_categories)):
                medicine_names.append(f"{category}_{i+1}")
        
        # Generate features
        manufacturing_years = np.random.randint(2020, 2025, n_samples)
        expiry_years = manufacturing_years + np.random.randint(1, 5, n_samples)
        
        # Price depends on manufacturing year, expiry year, and random factors
        base_prices = np.random.uniform(5, 200, n_samples)
        year_factor = (expiry_years - manufacturing_years) * 10
        price_noise = np.random.normal(0, 20, n_samples)
        prices = base_prices + year_factor + price_noise
        prices = np.maximum(prices, 1)  # Ensure positive prices
        
        # Quantity depends on price, expiry, and demand factors
        demand_factor = 1000 / (prices + 1)  # Higher price = lower demand
        expiry_factor = np.random.uniform(0.8, 1.2, n_samples)
        quantities = np.random.poisson(demand_factor * expiry_factor)
        quantities = np.maximum(quantities, 1)  # Ensure positive quantities
        
        # Create DataFrame
        data = pd.DataFrame({
            'medicine_name': medicine_names,
            'manufacturing_year': manufacturing_years,
            'expiry_year': expiry_years,
            'price': prices,
            'quantity': quantities,
            'days_to_expiry': (expiry_years - manufacturing_years) * 365,
            'price_category': pd.cut(prices, bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        })
        
        self.training_data = data
        return data
    
    def train_models(self):
        """Train all machine learning models"""
        if self.training_data is None:
            self.generate_training_data()
        
        data = self.training_data
        
        # Prepare features for price prediction
        X_price = data[['manufacturing_year', 'expiry_year', 'days_to_expiry', 'quantity']].values
        y_price = data['price'].values
        
        # Prepare features for quantity prediction
        X_quantity = data[['manufacturing_year', 'expiry_year', 'days_to_expiry', 'price']].values
        y_quantity = data['quantity'].values
        
        # Split data
        X_price_train, X_price_test, y_price_train, y_price_test = train_test_split(
            X_price, y_price, test_size=0.2, random_state=42
        )
        
        X_quantity_train, X_quantity_test, y_quantity_train, y_quantity_test = train_test_split(
            X_quantity, y_quantity, test_size=0.2, random_state=42
        )
        
        # Train Linear Regression for price
        lr_price = LinearRegression()
        lr_price.fit(X_price_train, y_price_train)
        self.models['linear_regression_price'] = lr_price
        
        # Train Polynomial Regression for price
        poly_features = PolynomialFeatures(degree=2)
        X_poly = poly_features.fit_transform(X_price_train)
        poly_reg = LinearRegression()
        poly_reg.fit(X_poly, y_price_train)
        self.models['polynomial_regression_price'] = (poly_features, poly_reg)
        
        # Train KNN for price
        knn_price = KNeighborsRegressor(n_neighbors=5)
        knn_price.fit(X_price_train, y_price_train)
        self.models['knn_price'] = knn_price
        
        # Train Decision Tree for price
        dt_price = DecisionTreeRegressor(random_state=42, max_depth=10)
        dt_price.fit(X_price_train, y_price_train)
        self.models['decision_tree_price'] = dt_price
        
        # Train models for quantity prediction
        lr_quantity = LinearRegression()
        lr_quantity.fit(X_quantity_train, y_quantity_train)
        self.models['linear_regression_quantity'] = lr_quantity
        
        knn_quantity = KNeighborsRegressor(n_neighbors=5)
        knn_quantity.fit(X_quantity_train, y_quantity_train)
        self.models['knn_quantity'] = knn_quantity
        
        self.is_trained = True
        
        # Calculate and return model performance
        performance = {}
        
        # Price prediction performance
        y_price_pred_lr = lr_price.predict(X_price_test)
        y_price_pred_poly = poly_reg.predict(poly_features.transform(X_price_test))
        y_price_pred_knn = knn_price.predict(X_price_test)
        y_price_pred_dt = dt_price.predict(X_price_test)
        
        performance['linear_regression_price'] = {
            'mse': mean_squared_error(y_price_test, y_price_pred_lr),
            'r2': r2_score(y_price_test, y_price_pred_lr)
        }
        performance['polynomial_regression_price'] = {
            'mse': mean_squared_error(y_price_test, y_price_pred_poly),
            'r2': r2_score(y_price_test, y_price_pred_poly)
        }
        performance['knn_price'] = {
            'mse': mean_squared_error(y_price_test, y_price_pred_knn),
            'r2': r2_score(y_price_test, y_price_pred_knn)
        }
        performance['decision_tree_price'] = {
            'mse': mean_squared_error(y_price_test, y_price_pred_dt),
            'r2': r2_score(y_price_test, y_price_pred_dt)
        }
        
        # Quantity prediction performance
        y_quantity_pred_lr = lr_quantity.predict(X_quantity_test)
        y_quantity_pred_knn = knn_quantity.predict(X_quantity_test)
        
        performance['linear_regression_quantity'] = {
            'mse': mean_squared_error(y_quantity_test, y_quantity_pred_lr),
            'r2': r2_score(y_quantity_test, y_quantity_pred_lr)
        }
        performance['knn_quantity'] = {
            'mse': mean_squared_error(y_quantity_test, y_quantity_pred_knn),
            'r2': r2_score(y_quantity_test, y_quantity_pred_knn)
        }
        
        return performance
    
    def predict_price(self, manufacturing_year, expiry_year, quantity, algorithm='linear_regression'):
        """Predict medicine price using specified algorithm"""
        if not self.is_trained:
            self.train_models()
        
        days_to_expiry = (expiry_year - manufacturing_year) * 365
        features = np.array([[manufacturing_year, expiry_year, days_to_expiry, quantity]])
        
        if algorithm == 'linear_regression':
            prediction = self.models['linear_regression_price'].predict(features)[0]
        elif algorithm == 'polynomial_regression':
            poly_features, poly_reg = self.models['polynomial_regression_price']
            features_poly = poly_features.transform(features)
            prediction = poly_reg.predict(features_poly)[0]
        elif algorithm == 'knn':
            prediction = self.models['knn_price'].predict(features)[0]
        elif algorithm == 'decision_tree':
            prediction = self.models['decision_tree_price'].predict(features)[0]
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        return max(0, prediction)  # Ensure non-negative price
    
    def predict_quantity(self, manufacturing_year, expiry_year, price, algorithm='linear_regression'):
        """Predict medicine quantity using specified algorithm"""
        if not self.is_trained:
            self.train_models()
        
        days_to_expiry = (expiry_year - manufacturing_year) * 365
        features = np.array([[manufacturing_year, expiry_year, days_to_expiry, price]])
        
        if algorithm == 'linear_regression':
            prediction = self.models['linear_regression_quantity'].predict(features)[0]
        elif algorithm == 'knn':
            prediction = self.models['knn_quantity'].predict(features)[0]
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        return max(1, int(prediction))  # Ensure positive integer quantity
    
    def get_model_comparison_chart(self):
        """Generate a comparison chart of all models"""
        if not self.is_trained:
            self.train_models()
        
        # Get model names and their R² scores
        model_names = []
        r2_scores = []
        
        for name, model in self.models.items():
            if 'price' in name:
                if name == 'polynomial_regression_price':
                    poly_features, poly_reg = model
                    y_pred = poly_reg.predict(poly_features.transform(
                        self.training_data[['manufacturing_year', 'expiry_year', 'days_to_expiry', 'quantity']].values
                    ))
                    r2 = r2_score(self.training_data['price'], y_pred)
                else:
                    y_pred = model.predict(
                        self.training_data[['manufacturing_year', 'expiry_year', 'days_to_expiry', 'quantity']].values
                    )
                    r2 = r2_score(self.training_data['price'], y_pred)
                
                model_names.append(name.replace('_price', '').replace('_', ' ').title())
                r2_scores.append(r2)
        
        # Create comparison chart
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, r2_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        plt.title('Model Performance Comparison (R² Score)', fontsize=16, fontweight='bold')
        plt.xlabel('Algorithm', fontsize=12)
        plt.ylabel('R² Score', fontsize=12)
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars, r2_scores):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Convert plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
        
        return base64.b64encode(image_png).decode()
    
    def get_price_distribution_chart(self):
        """Generate price distribution chart"""
        if self.training_data is None:
            self.generate_training_data()
        
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.training_data, x='price', bins=30, color='#FF6B6B', alpha=0.7)
        plt.title('Medicine Price Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Price (₹)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.axvline(self.training_data['price'].mean(), color='red', linestyle='--', 
                    label=f'Mean: ₹{self.training_data["price"].mean():.2f}')
        plt.legend()
        plt.tight_layout()
        
        # Convert plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
        
        return base64.b64encode(image_png).decode()
    
    def get_quantity_vs_price_chart(self):
        """Generate quantity vs price scatter plot"""
        if self.training_data is None:
            self.generate_training_data()
        
        plt.figure(figsize=(10, 6))
        plt.scatter(self.training_data['price'], self.training_data['quantity'], 
                   alpha=0.6, color='#4ECDC4', s=50)
        plt.title('Quantity vs Price Relationship', fontsize=16, fontweight='bold')
        plt.xlabel('Price (₹)', fontsize=12)
        plt.ylabel('Quantity', fontsize=12)
        
        # Add trend line
        z = np.polyfit(self.training_data['price'], self.training_data['quantity'], 1)
        p = np.poly1d(z)
        plt.plot(self.training_data['price'], p(self.training_data['price']), 
                "r--", alpha=0.8, linewidth=2)
        
        plt.tight_layout()
        
        # Convert plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
        
        return base64.b64encode(image_png).decode()

# Global instance
ml_predictor = MLPredictor()


class MedicineSubstitutionRecommender:

    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), min_df=1) if TfidfVectorizer else None
        self._matrix = None
        self._id_to_medicine: list[Medicine] = []
        self._molecule_sets: list[set[str]] = []
        self._embeddings = None

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = (text or '').lower()
        text = re.sub(r"[^a-z0-9\s\+/,()-]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _parse_molecules(text: str) -> set[str]:
        if not text:
            return set()
        text = text.lower()
        text = re.sub(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|%|iu)\b", " ", text)
        tokens = re.split(r"[\s\+/,()\-]+", text)
        stop = {"tablet","tablets","capsule","capsules","syrup","injection","oral","suspension","extended","release"}
        tokens = [t for t in tokens if len(t) > 1 and t not in stop]
        return set(tokens)

    def _build_corpus(self, medicines: QuerySet[Medicine] | list[Medicine]):
        self._id_to_medicine = list(medicines)
        texts = []
        molecule_sets = []
        for med in self._id_to_medicine:
            text = (med.composition or '').strip() or (med.m_descr or '').strip()
            texts.append(self._normalize_text(text))
            molecule_sets.append(self._parse_molecules(med.composition))
        self._molecule_sets = molecule_sets
        if self.vectorizer:
            self._matrix = self.vectorizer.fit_transform(texts)
        # Semantic embeddings if available
        global _st_model
        if SentenceTransformer and _st_model is None:
            try:
                _st_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            except Exception:
                _st_model = None
        if _st_model is not None:
            try:
                self._embeddings = _st_model.encode(texts, normalize_embeddings=True)
            except Exception:
                self._embeddings = None

    def recommend_substitutes(self, medicine: Medicine, top_k: int = 5):
        medicines = Medicine.objects.exclude(id=medicine.id).filter(m_quantity__gt=0)
        if not medicines.exists() or not self.vectorizer or cosine_similarity is None:
            return []
        self._build_corpus(medicines)
        query_text = self._normalize_text((medicine.composition or medicine.m_descr or '').strip())
        if not query_text:
            return []
        query_vec = self.vectorizer.transform([query_text])
        sims = cosine_similarity(query_vec, self._matrix).flatten()
        # Jaccard on normalized molecule sets
        query_mols = self._parse_molecules(medicine.composition)
        jaccs = []
        for mols in self._molecule_sets:
            denom = len(query_mols | mols)
            j = (len(query_mols & mols) / denom) if denom else 0.0
            jaccs.append(j)
        # Semantic similarity if available
        sem_sims = None
        if self._embeddings is not None:
            try:
                q_emb = _st_model.encode([query_text], normalize_embeddings=True)
                sem_sims = cosine_similarity(q_emb, self._embeddings).flatten()
            except Exception:
                sem_sims = None
        results_scored = []
        for idx, med in enumerate(self._id_to_medicine):
            tfidf_sim = float(sims[idx])
            jacc = float(jaccs[idx])
            sem = float(sem_sims[idx]) if sem_sims is not None else 0.0
            combined_sim = 0.5*tfidf_sim + 0.3*jacc + 0.2*sem
            try:
                price_saving = max(0.0, float(medicine.m_price) - float(med.m_price))
                saving_norm = price_saving / (float(medicine.m_price) + 1e-6)
            except Exception:
                saving_norm = 0.0
            stock_norm = min(1.0, float(med.m_quantity) / 100.0) if med.m_quantity is not None else 0.0
            expiry_penalty = 0.2 if (med.m_edate and (med.m_edate - date.today()).days <= 90) else 0.0
            generic_bonus = 0.05 if med.is_generic else 0.0
            final_score = combined_sim + 0.15*saving_norm + 0.05*stock_norm + generic_bonus - expiry_penalty
            results_scored.append((med, final_score))
        results_scored.sort(key=lambda x: x[1], reverse=True)
        results = []
        for med, score in results_scored[:top_k]:
            results.append({
                'name': med.m_name,
                'price': float(med.m_price),
                'composition': med.composition,
                'similarity': float(score),
                'is_generic': med.is_generic,
                'id': med.id,
            })
        return results


class DiseaseToMedicineRecommender:
    """TF-IDF + keyword overlap + optional semantic embeddings recommender."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), min_df=1) if TfidfVectorizer else None
        self._matrix = None
        self._id_to_medicine: list[Medicine] = []
        self._embeddings = None
        self._indication_tokens: list[set[str]] = []

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = (text or '').lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        text = DiseaseToMedicineRecommender._normalize_text(text)
        tokens = set(t for t in text.split() if len(t) > 2)
        return tokens

    def _fit(self):
        medicines = Medicine.objects.filter(m_quantity__gt=0)
        self._id_to_medicine = list(medicines)
        texts = [self._normalize_text((m.indications or m.m_descr or '').strip()) for m in self._id_to_medicine]
        self._indication_tokens = [self._tokenize(m.indications or m.m_descr or '') for m in self._id_to_medicine]
        if not self.vectorizer:
            return False
        self._matrix = self.vectorizer.fit_transform(texts)
        global _st_model
        if SentenceTransformer and _st_model is None:
            try:
                _st_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            except Exception:
                _st_model = None
        if _st_model is not None:
            try:
                self._embeddings = _st_model.encode(texts, normalize_embeddings=True)
            except Exception:
                self._embeddings = None
        return True

    def recommend(self, disease_text: str, top_k: int = 5):
        if not disease_text or not self.vectorizer:
            return []
        if self._matrix is None or not self._id_to_medicine:
            if not self._fit():
                return []
        qtext = self._normalize_text(disease_text)
        query_vec = self.vectorizer.transform([qtext])
        sims = cosine_similarity(query_vec, self._matrix).flatten()
        q_tokens = self._tokenize(qtext)
        jaccs = []
        for toks in self._indication_tokens:
            denom = len(q_tokens | toks)
            j = (len(q_tokens & toks) / denom) if denom else 0.0
            jaccs.append(j)
        sem_sims = None
        if self._embeddings is not None:
            try:
                q_emb = _st_model.encode([qtext], normalize_embeddings=True)
                sem_sims = cosine_similarity(q_emb, self._embeddings).flatten()
            except Exception:
                sem_sims = None
        results_scored = []
        for idx, med in enumerate(self._id_to_medicine):
            tfidf_sim = float(sims[idx])
            jacc = float(jaccs[idx])
            sem = float(sem_sims[idx]) if sem_sims is not None else 0.0
            combined_sim = 0.6*tfidf_sim + 0.2*jacc + 0.2*sem
            generic_bonus = 0.03 if med.is_generic else 0.0
            final_score = combined_sim + generic_bonus
            results_scored.append((med, final_score))
        results_scored.sort(key=lambda x: x[1], reverse=True)
        top = []
        for med, score in results_scored[:top_k]:
            top.append({
                'name': med.m_name,
                'price': float(med.m_price),
                'indications': med.indications,
                'similarity': float(score),
                'is_generic': med.is_generic,
                'id': med.id,
            })
        return top


# Global instances
substitution_recommender = MedicineSubstitutionRecommender()
disease_recommender = DiseaseToMedicineRecommender()
