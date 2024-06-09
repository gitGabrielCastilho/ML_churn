import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE


def gerar_base():
    num_records = 10000
    #np.random.seed(1)

    customer_ids = np.arange(1, num_records + 1)
    ages = np.random.normal(loc=40, scale=10, size=num_records).astype(int) 
    genders = np.random.choice(['Male', 'Female'], size=num_records, p=[0.5, 0.5])
    tenures = np.random.poisson(lam=3, size=num_records)  
    balances = np.random.normal(loc=50000, scale=20000, size=num_records)  
    num_of_products = np.random.choice([1, 2, 3, 4], size=num_records, p=[0.5, 0.3, 0.15, 0.05])
    has_cr_card = np.random.choice([0, 1], size=num_records, p=[0.3, 0.7])
    is_active_member = np.random.choice([0, 1], size=num_records, p=[0.7, 0.3])
    estimated_salaries = np.random.normal(loc=100000, scale=40000, size=num_records)  

 
    churn_prob = (
        0.4 * (ages > 60) +  
        0.3 * (balances < 20000) +  
        0.2 * (num_of_products == 1) +  
        0.3 * (is_active_member == 0) +  
        0.1 * (tenures < 1) +  
        0.1 * (estimated_salaries < 50000)  
    )

    random_churn = np.random.rand(num_records) < 0.1 
    churn = ((churn_prob + np.random.rand(num_records)) > 0.5) | random_churn
    churn = churn.astype(int)

    df = pd.DataFrame({
        'CustomerID': customer_ids,
        'Age': ages,
        'Gender': genders,
        'Tenure': tenures,
        'Balance': balances,
        'NumOfProducts': num_of_products,
        'HasCrCard': has_cr_card,
        'IsActiveMember': is_active_member,
        'EstimatedSalary': estimated_salaries,
        'Churn': churn
    })

    df['Age'] = df['Age'].clip(18, 90)
    df['Balance'] = df['Balance'].clip(0, None)
    df['EstimatedSalary'] = df['EstimatedSalary'].clip(20000, None)

    df.to_csv('machine-learning-churn/bank_churn.csv', index=False)


def predict_churn():
    df = pd.read_csv('machine-learning-churn/bank_churn.csv')
    df = df.drop(columns=['CustomerID'])
    df = pd.get_dummies(df, drop_first=True)


    x = df.drop(columns=['Churn'])
    y = df['Churn']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3) #, random_state=

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    smote = SMOTE() #random_state=1
    x_train_res, y_train_res = smote.fit_resample(x_train, y_train)

    model = RandomForestClassifier() #random_state=1

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True]
    }

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(x_train_res, y_train_res)
    print(grid_search.best_params_)

    best_model = grid_search.best_estimator_
    best_model.fit(x_train_res, y_train_res)

    y_pred = best_model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Acurácia: {accuracy * 100:.2f}%')
    print(classification_report(y_test, y_pred))

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f'Precisão: {precision * 100:.2f}%')
    print(f'Recall: {recall * 100:.2f}%')
    print(f'F1-Score: {f1 * 100:.2f}%')

##########################################################################################

    # Visualização dos Resultados
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    feature_importances = best_model.feature_importances_
    features = x.columns
    plt.figure(figsize=(13, 6))
    plt.barh(features, feature_importances)
    plt.xlabel('Importância das Features')
    plt.ylabel('Features')
    plt.title('Importância das Features no Modelo Random Forest')
    plt.show()

if __name__ == "__main__":
    gerar_base()
    predict_churn()