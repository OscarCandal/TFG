import sys
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.ensemble import IsolationForest
from pyspark.sql import SparkSession
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,confusion_matrix,roc_auc_score
import pyspark.ml.classification
import sklearn.linear_model
from pyspark.ml.feature import VectorAssembler
import pyspark.ml.feature
import sklearn.preprocessing
import sklearn.ensemble
import sklearn.neural_network
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier

#Revisar que estea ben esto
np_random_seed = 2
np.random.seed(2)

## Obtencion de los datasets por separado
dataset_ovs=pd.read_csv('/home/oscar.candal.suarez/dataset/OVS.csv')
dataset_metasploit=pd.read_csv('/home/oscar.candal.suarez/dataset/metasploitable-2.csv')
dataset_normalData=pd.read_csv(r'/home/oscar.candal.suarez/dataset/Normal_data.csv')

## Fusion de los 3 datasets en 1 solo
dataset = [dataset_metasploit, dataset_normalData, dataset_ovs]
dataset_total = pd.concat(dataset, ignore_index=True)

## Cambiamos la feature "Label" para darle un valor numerico. 
## 1 para el tráfico de ataque
## 0 para el tráfico normal
cleanup_nums = {"Label": {"Normal":0,"Probe":1,"DDoS":1,"DoS":1,"BFA":1,"Web-Attack":1,"BOTNET":1,"U2R":1,"DDoS ":1}}
dataset_total_filtrado = dataset_total.replace(cleanup_nums)

## Eliminacion de las features cuyo valor no variaba de una fila a otra y del identificador (valores de NaN en la matriz de correlación)
dt_not_NaN= dataset_total_filtrado.drop(['Flow ID','Fwd PSH Flags','Fwd URG Flags','CWE Flag Count','ECE Flag Cnt','Fwd Byts/b Avg','Fwd Pkts/b Avg','Fwd Blk Rate Avg','Bwd Byts/b Avg','Bwd Pkts/b Avg','Bwd Blk Rate Avg','Init Fwd Win Byts','Fwd Seg Size Min'], axis=1)

## Creación de variables para conservar las características cíclicas de los atributos
dt_new = dt_not_NaN

dt_new['hr_sin_minutes'] = np.sin(pd.to_datetime(dt_new.Timestamp).dt.minute*(2.*np.pi/60))
dt_new['hr_cos_minutes'] = np.cos(pd.to_datetime(dt_new.Timestamp).dt.minute*(2.*np.pi/60))
dt_new['hr_sin_hours'] = np.sin(pd.to_datetime(dt_new.Timestamp).dt.hour*(2.*np.pi/24))
dt_new['hr_cos_hours'] = np.cos(pd.to_datetime(dt_new.Timestamp).dt.hour*(2.*np.pi/24))

#Creación de variable para ver si es fin de semana o no
dt_new['Weekend'] = pd.to_datetime(dt_new.Timestamp,dayfirst=True).dt.weekday
dt_new['Weekend'] = (dt_new['Weekend'] > 4).astype(int)

#Creación de variables en función de la subred de destino
dt_new['Dst 192.168.8']=np.where(dt_new["Dst IP"].str.startswith('192.168.8.'),1,0)
dt_new['Dst 192.168.3.']=np.where(dt_new["Dst IP"].str.startswith('192.168.3.'),1,0)
dt_new['Dst 200.175.2.']=np.where(dt_new["Dst IP"].str.startswith('200.175.2.'),1,0)
dt_new['Dst 192.168.20.']=np.where(dt_new["Dst IP"].str.startswith('192.168.20.'),1,0)
dt_new['Dst 172.17.0.']=np.where(dt_new["Dst IP"].str.startswith('172.17.0.'),1,0)

#Creación de variables en función de la subred de fuente
dt_new['Src 192.168.3.']=np.where(dt_new["Src IP"].str.startswith('192.168.3.'),1,0)
dt_new['Src 200.175.2.']=np.where(dt_new["Src IP"].str.startswith('200.175.2.'),1,0)
dt_new['Src 192.168.20.']=np.where(dt_new["Src IP"].str.startswith('192.168.20.'),1,0)
dt_new['Src 172.17.0.']=np.where(dt_new["Src IP"].str.startswith('172.17.0.'),1,0)

#Limpeza de variables que no se van a usar
dt_new= dt_new.drop(['Src IP','Dst IP','Timestamp'], axis=1)

#Limpieza de valores negativos en el campo de Flow Duration
negative_index = dt_new.index[dt_new['Flow Duration'] < 0]
dt_new = dt_new.drop(negative_index)

#Isolation_Forest

#Casos a eliminar
#1 Variable 1 -> Valores de 'Flow Duration' demasiado altos
#2 Variable 2 -> Valores de 'Flow IAT Std', 'Fwd IAT Std', 'Flow IAT Max' y 'Fwd IAT Tot' demasiado altos
#3 Variable 3 -> Valores de 'Bwd IAT Tot' muy altos
#4 Variable 4 -> Valores de 'Pkt Len Max' muy altos

dt_new_delete_outliers = dt_new

variable1=['Flow Duration', 'Tot Fwd Pkts','Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max','Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std','Bwd Pkt Len Max', 'Bwd Pkt Len Min']
variable2=['Bwd Pkt Len Mean','Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean','Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot','Fwd IAT Mean', 'Fwd IAT Std']
variable3=['Fwd IAT Max', 'Fwd IAT Min','Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max','Bwd IAT Min', 'Bwd PSH Flags', 'Bwd URG Flags', 'Fwd Header Len','Bwd Header Len']
variable4=['Fwd Pkts/s', 'Bwd Pkts/s', 'Pkt Len Min','Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var','FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt']
variable5=['ACK Flag Cnt', 'URG Flag Cnt', 'Down/Up Ratio', 'Pkt Size Avg','Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Subflow Fwd Pkts','Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts','Init Bwd Win Byts']
variable6=['Fwd Act Data Pkts', 'Active Mean', 'Active Std','Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max','Idle Min']

model=IsolationForest(n_estimators=50, max_samples='auto', contamination=0.0001,max_features=8, random_state=1)

model.fit(dt_new_delete_outliers[variable1])
dt_new_delete_outliers['anomaly']=model.predict(dt_new_delete_outliers[variable1])
anomaly=dt_new_delete_outliers.loc[dt_new_delete_outliers['anomaly']==-1]
anomaly_index=list(anomaly.index)
dt_new = dt_new.drop(anomaly_index)

model.fit(dt_new_delete_outliers[variable2])
dt_new_delete_outliers['anomaly']=model.predict(dt_new_delete_outliers[variable2])
anomaly=dt_new_delete_outliers.loc[dt_new_delete_outliers['anomaly']==-1]
anomaly_index=list(anomaly.index)
dt_new = dt_new.drop(anomaly_index)

model.fit(dt_new_delete_outliers[variable3])
dt_new_delete_outliers['anomaly']=model.predict(dt_new_delete_outliers[variable3])
anomaly=dt_new_delete_outliers.loc[dt_new_delete_outliers['anomaly']==-1]
anomaly_index=list(anomaly.index)
dt_new = dt_new.drop(anomaly_index)

model.fit(dt_new_delete_outliers[variable4])
dt_new_delete_outliers['anomaly']=model.predict(dt_new_delete_outliers[variable4])
anomaly=dt_new_delete_outliers.loc[dt_new_delete_outliers['anomaly']==-1]
anomaly_index=list(anomaly.index)
dt_new = dt_new.drop(anomaly_index)

dt_new= dt_new.drop(['anomaly'], axis=1)

#FIN-Isolation_Forest

#Alternativa rápida a Isolation Forest
#eliminados1= [137634, 138049, 147620, 147664, 147667, 147864, 148224, 148677, 148919, 161840, 161850, 162344, 162484, 162553, 163620, 165565, 170556, 171865, 172068, 172073, 172372, 172887, 173297, 174005, 174117, 174122, 174772, 174882, 183635, 184528, 185984, 195031, 202947, 202972]
#eliminados2= [138429, 138540, 138768, 140200, 140343, 140449, 142446, 148192, 148802, 149547, 154315, 155973, 156111, 157924, 162540, 168084, 169795, 170457, 170502, 172539, 174147, 174305, 178548, 186159, 186622, 192352, 199391, 204259, 275986, 276453, 278726, 279174, 281082]
#eliminados3= [148217, 149500, 149601, 152625, 153470, 153924, 154777, 154869, 154989, 155226, 155270, 155595, 155913, 173338, 183233]
#eliminados4= [142919, 149492, 162537, 167153, 167785, 167854, 168314, 168498, 168674, 168866, 169467, 169601, 169697, 169907, 170153, 172172, 173961, 183184, 187781, 195726, 195774, 197408, 201427, 204487, 204554]
#dt_new = dt_new.drop(eliminados1)
#dt_new = dt_new.drop(eliminados2)
#dt_new = dt_new.drop(eliminados3)
#dt_new = dt_new.drop(eliminados4)


#Variables con correlación muy elevada

#Flow Duration -> Duración del flow en Microsegundos
#------Bwd IAT Tot - 0,98 -> Total del t. entre 2 paquetes enviados en el flow dir backward
#Se elimina esta por que tiene menos correlación con la label 0.01
dt_new= dt_new.drop(['Bwd IAT Tot'], axis=1)


#Tot Fwd Pkts -> Número de paquetes en la dirección forward
#------Subflow Fwd Pkts - 1.00 -> Número medio de paquetes en sub flow en d. forward
#Indiferente mismo valor
dt_new= dt_new.drop(['Subflow Fwd Pkts'], axis=1)


#TotLen Fwd Pkts -> Tamaño total de los paquetes en la dirección forward
#------Subflow Fwd Byts - 0.99 -> Número medio de Bytes en sub flow en d. forward
#Indiferente mismo valor
dt_new= dt_new.drop(['Subflow Fwd Byts'], axis=1)


#TotLen Bwd Pkts -> Tamaño total de los paquetes en la dirección backward
#------ Subflow Bwd Byts - 0.99  -> Número medio de Bytes en sub flow en d. backward
#Indiferente mismo valor
dt_new= dt_new.drop(['Subflow Bwd Byts'], axis=1)


#Tot Bwd Pkts -> Número de paquetes en la dirección backward
#------ Bwd Header Len - 0.99 -> Número de bytes usados para cabeceras en la dir backward
#------Subflow Bwd Pkts - 1.00 -> Número medio de paquetes en sub flow en d. backward
dt_new= dt_new.drop(['Bwd Header Len'], axis=1)
dt_new= dt_new.drop(['Subflow Bwd Pkts'], axis=1)


#Fwd Pkt Len Mean -> Tamaño medio de paquete en dir forward
#------ Fwd Pkt Len Std - 0.954 -> Desviación estándar del tamaño de paquete en dir forward
#------ Fwd Seg Size Avg - 1.00 -> Tamaño medio observado en la dirección de forward
dt_new= dt_new.drop(['Fwd Pkt Len Std'], axis=1)
dt_new= dt_new.drop(['Fwd Seg Size Avg'], axis=1)


# Bwd Pkt Len Max -> Tamaño máximo de paquete en dir backward
#------ Pkt Len Max  - 0.97 -> Máximo del tamaño de un paquete
dt_new= dt_new.drop(['Bwd Pkt Len Max'], axis=1)


# Bwd Pkt Len Mean -> Tamaño medio de paquete en dir backward
#------Bwd Seg Size Avg - 1.0000 -> Número medio de Bts bulk rate observado en la d. de backward
dt_new= dt_new.drop(['Bwd Seg Size Avg'], axis=1)


# Flow Pkts/s -> Flow paquetes por segundo
#------ Bwd Pkts/s - 0.99 -> Número de paquetes backward por segundo 
dt_new= dt_new.drop(['Flow Pkts/s'], axis=1)


#Flow IAT Std -> Desviación estándar del tiempo entre 2 paquetes enviados en el flow
#------ Flow IAT Max - 0.98 -> Tiempo máximo entre 2 paquetes enviados en el flow
#------ Bwd IAT Std - 0.98 -> Tiempo medio entre 2 paquetes enviados en el flow dir backward
#------ Bwd IAT Max - 0.97 -> Tiempo máximo entre 2 paquetes enviados en el flow dir backward
#------ Idle Mean - 0.98 -> Tiempo media que el flujo estuvo inactivo antes estar activo
#------ Idle Max - 0.98 -> Tiempo máximo que el flujo estuvo inactivo antes estar activo
#------ Idle Min - 0.98 ->Tiempo  mínimo que el flujo estuvo inactivo antes estar activo

dt_new= dt_new.drop(['Flow IAT Std'], axis=1)
dt_new= dt_new.drop(['Flow IAT Max'], axis=1)
dt_new= dt_new.drop(['Bwd IAT Max'], axis=1)
dt_new= dt_new.drop(['Idle Mean'], axis=1)
dt_new= dt_new.drop(['Idle Max'], axis=1)
dt_new= dt_new.drop(['Idle Min'], axis=1)


# Bwd PSH Flags -> Número de veces que el flag PSH usado en paquetes dir backward
#------ PSH Flag Cnt - 1 -> Número de paquetes con el flag PSH 
dt_new= dt_new.drop(['Bwd PSH Flags'], axis=1)


# Bwd URG Flags -> Número de veces que el flag URG usado en paquetes dir backward
#------ URG Flag Cnt - 1 -> Número de paquetes con el flag URG 
dt_new= dt_new.drop(['URG Flag Cnt'], axis=1)


# Pkt Len Mean -> Media del tamaño de un paquete
#------ Pkt Len Std - 0. 96 -> Desviación estándar del tamaño de un paquete
#------ Pkt Size Avg - 0.99 -> Tamaño medio del paquete
dt_new= dt_new.drop(['Pkt Len Mean'], axis=1)
dt_new= dt_new.drop(['Pkt Len Std'], axis=1)
print(dt_new.shape)


#RFECV
target = dt_new['Label']
X = dt_new.drop('Label', axis=1)
rfc = DecisionTreeClassifier(random_state=0)

#Ejecucion de RFECV
rfecv = RFECV(estimator=rfc, step=1, cv=StratifiedKFold(n_splits=5, shuffle=True,random_state=3), scoring='accuracy',verbose=4, min_features_to_select=1)
rfecv.fit(X, target)

#Obtener el numero de caracteristicas son optimas para producir el mejor scoring selecionado 
print('Optimal number of features: {}'.format(rfecv.n_features_))
print("Características a no seleccionar")
print(np.where(rfecv.support_ == False)[0])
print("Características a seleccionar")
print(np.where(rfecv.support_ == True)[0])
X.drop(X.columns[np.where(rfecv.support_ == False)[0]], axis=1, inplace=True)
selected_columns = X.columns.tolist()
selected_columns.append('Label')
dt_new = dt_new[selected_columns]
#FIN-RFECV

#Eliminacion de atributos alternativa a aplicar RFECV, eliminación para obtener los 28 atributos (teniendo en cuenta la label)
#dt_new= dt_new.drop(['Src Port','Protocol','Flow Duration','Tot Fwd Pkts','Fwd Pkt Len Mean', 
#'Bwd Pkt Len Min','Bwd Pkt Len Mean','Flow Byts/s','Flow IAT Mean','Fwd IAT Tot','Fwd IAT Std',
#'Fwd IAT Max','Bwd IAT Mean','Bwd IAT Std','Bwd IAT Min','Bwd URG Flags','Fwd Header Len','Fwd Pkts/s',
#'Pkt Len Min','Pkt Len Var','FIN Flag Cnt','SYN Flag Cnt','RST Flag Cnt','PSH Flag Cnt','ACK Flag Cnt',
#'Pkt Size Avg','Active Mean','Active Std','Active Max','Weekend','Dst 192.168.8','Dst 172.17.0.','Src 172.17.0.'], axis=1)

#Se hace el reseteo de los indices para poder trabajar con ellos
dt_new.reset_index(inplace=True,drop=True)

X = dt_new.drop('Label', axis=1)
y = dt_new['Label']

param_grids = [
{
    'LogisticRegression__penalty': ['l2'],
    'LogisticRegression__C': [20,30,40,50]
},{ 
    'RF_Classifier__n_estimators': [100, 200, 300],
    'RF_Classifier__max_features': ['sqrt']
},{
    'MLPClassifier__hidden_layer_sizes': [(50,50,50), (50,50), (100,)]
}]

pipes = [
    Pipeline([("scaler", sklearn.preprocessing.StandardScaler()), ("LogisticRegression", sklearn.linear_model.LogisticRegression(max_iter=20000,random_state=1))]),
    Pipeline([("RF_Classifier",sklearn.ensemble.RandomForestClassifier(random_state=1))]),
    Pipeline([("scaler", sklearn.preprocessing.StandardScaler()), ("MLPClassifier", sklearn.neural_network.MLPClassifier(random_state=1))])
]

names = ["LogisticRegression","RF_Classifier","MLPClassifier"]

#----------------Parte distribuida----------------
spark = SparkSession.builder.getOrCreate()
old_value=spark.conf.get("spark.sql.autoBroadcastJoinThreshold")
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)

#Preparación de los datos de entrada
#Concentración de todas las variables excepto la label (27 features)
assembler = VectorAssembler(
inputCols=['Dst Port', 'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts',
    'Fwd Pkt Len Max', 'Fwd Pkt Len Min', 'Bwd Pkt Len Std', 'Flow IAT Min',
    'Fwd IAT Mean', 'Fwd IAT Min', 'Bwd Pkts/s', 'Pkt Len Max',
    'Down/Up Ratio', 'Init Bwd Win Byts', 'Fwd Act Data Pkts', 'Active Min',
    'Idle Std', 'hr_sin_minutes', 'hr_cos_minutes', 'hr_sin_hours',
    'hr_cos_hours', 'Dst 192_168_3_', 'Dst 200_175_2_','Dst 192_168_20_',
    'Src 192_168_3_', 'Src 200_175_2_', 'Src 192_168_20_'],
outputCol="features")

for param_grid, pipe, name in zip(param_grids, pipes, names):
    accuracy_array = []
    precision_array = []
    recall_array = []
    f1__score_array = []
    confusion__matrix_array = []
    time_array = []
    best_params_array = []

    cv_externo = StratifiedKFold(n_splits=5, shuffle=True,random_state=1)

    for train_index, test_index in cv_externo.split(X,y):
        X_train,X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train,y_test = y.iloc[train_index], y.iloc[test_index]

        X_train_to_distribute = dt_new.iloc[train_index,:]
        X_test_to_distribute = dt_new.iloc[test_index,:]

        grid = GridSearchCV(pipe, param_grid=param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True,random_state=1),verbose=4)
        grid.fit(X_train, y_train)

        print("Best cross-validation accuracy: {:.9f}".format(grid.best_score_))
        print("Test set score: {:.9f}".format(grid.score(X_test, y_test)))
        print("Best parameters: {}".format(grid.best_params_))
        best_params_array.append(grid.best_params_)
        
        parametros= grid.cv_results_['params']
        i=0
        a=0
        for parametro in parametros:
            if parametro == grid.best_params_:
                a=i
            i=i+1
        time_array.append(grid.cv_results_['mean_fit_time'][a])

        #----------------Parte distribuida-Preparacion----------------

        if (name=="LogisticRegression"):
            best_param_c=grid.best_params_['LogisticRegression__C']
            best_param_reg_param= 1/best_param_c
            lr = pyspark.ml.classification.LogisticRegression(featuresCol = 'scaled_features', labelCol = 'Label', maxIter=100, elasticNetParam = 0, regParam = best_param_reg_param)

        elif(name=="RF_Classifier"):
            best_param_n_estimators = grid.best_params_['RF_Classifier__n_estimators']
            lr = pyspark.ml.classification.RandomForestClassifier(featuresCol = 'features', labelCol = 'Label',featureSubsetStrategy = "sqrt" ,numTrees=best_param_n_estimators)

        else:
            best_param_hidden_layer_sizes = list(grid.best_params_['MLPClassifier__hidden_layer_sizes'])
            best_param_hidden_layer_sizes.insert(0,27)
            best_param_hidden_layer_sizes.insert(len(best_param_hidden_layer_sizes),2)
            print(best_param_hidden_layer_sizes)
            lr = pyspark.ml.classification.MultilayerPerceptronClassifier(featuresCol = 'scaled_features', labelCol = 'Label', layers= best_param_hidden_layer_sizes)


        sparkDF_train=spark.createDataFrame(X_train_to_distribute)
        sparkDF_test=spark.createDataFrame(X_test_to_distribute)

        #Cambio de nombre de las variables con IPs, los nombres con puntos
        #No son validos en el entorno distribuido
        sparkDF_train=sparkDF_train.withColumnRenamed("Dst 192.168.3.","Dst 192_168_3_")
        sparkDF_train=sparkDF_train.withColumnRenamed("Dst 200.175.2.","Dst 200_175_2_")

        sparkDF_train=sparkDF_train.withColumnRenamed("Src 192.168.3.","Src 192_168_3_")
        sparkDF_train=sparkDF_train.withColumnRenamed("Src 200.175.2.","Src 200_175_2_")

        sparkDF_train=sparkDF_train.withColumnRenamed("Src 192.168.20.","Src 192_168_20_")
        sparkDF_train=sparkDF_train.withColumnRenamed("Dst 192.168.20.","Dst 192_168_20_")

        sparkDF_test=sparkDF_test.withColumnRenamed("Dst 192.168.3.","Dst 192_168_3_")
        sparkDF_test=sparkDF_test.withColumnRenamed("Dst 200.175.2.","Dst 200_175_2_")

        sparkDF_test=sparkDF_test.withColumnRenamed("Src 192.168.3.","Src 192_168_3_")
        sparkDF_test=sparkDF_test.withColumnRenamed("Src 200.175.2.","Src 200_175_2_")

        sparkDF_test=sparkDF_test.withColumnRenamed("Src 192.168.20.","Src 192_168_20_")
        sparkDF_test=sparkDF_test.withColumnRenamed("Dst 192.168.20.","Dst 192_168_20_")

        #Obtención del Dataset de tipo: pyspark.sql.dataframe.DataFrame
        sparkDF_train = assembler.transform(sparkDF_train)
        print("Shape de sparkDF_train")
        print((sparkDF_train.count(), len(sparkDF_train.columns)))

        sparkDF_test = assembler.transform(sparkDF_test)
        print("Shape de sparkDF_test")
        print((sparkDF_test.count(), len(sparkDF_test.columns)))

        #Realizamos el escalado en el entorno local 
        if (name!="RF_Classifier"):
            scaler = pyspark.ml.feature.StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=False)
            scalerModel = scaler.fit(sparkDF_train)
            sparkDF_train = scalerModel.transform(sparkDF_train)
            sparkDF_test = scalerModel.transform(sparkDF_test)

        ##Creacion del modelo
        lrModel= lr.fit(sparkDF_train)

        predictionAndTarget = lrModel.transform(sparkDF_test).select("Label", "prediction")
        predictionAndTargetNumpy = np.array((predictionAndTarget.collect()))

        acc = accuracy_score(predictionAndTargetNumpy[:,0], predictionAndTargetNumpy[:,1])
        print("Acc:")
        print(acc)
        accuracy_array.append(acc)

        f1 = f1_score(predictionAndTargetNumpy[:,0], predictionAndTargetNumpy[:,1])
        print("F1:")
        print(f1)
        f1__score_array.append(f1)

        precision = precision_score(predictionAndTargetNumpy[:,0], predictionAndTargetNumpy[:,1])
        print("Precision:")
        print(precision)
        precision_array.append(precision)

        recall = recall_score(predictionAndTargetNumpy[:,0], predictionAndTargetNumpy[:,1])
        print("Recall:")
        print(recall)
        recall_array.append(recall)

        auc = roc_auc_score(predictionAndTargetNumpy[:,0], predictionAndTargetNumpy[:,1])
        print("Auc:")
        print(auc)

    f = open('resultados_final.txt','a')
    default_sysout = sys.stdout
    sys.stdout = f
    print(name)
    print()
    print("Los mejores parametros selecionados son los siguientes")
    for param in best_params_array:
        print(param)
        print()
    print()
    print("Metricas finales:")
    print("Accuracy: ",accuracy_array)
    print("Accuracy mean:",sum(accuracy_array)/len(accuracy_array))
    print()
    print("Precision: ",precision_array)
    print("Precision mean:",sum(precision_array)/len(precision_array))
    print()
    print("Recall: ",recall_array)
    print("Recall mean:",sum(recall_array)/len(recall_array))
    print()
    print("F1_score:",f1__score_array)
    print("F1_score mean:",sum(f1__score_array)/len(f1__score_array))
    print()
    print("Training time: ",time_array)
    print("Time mean:",sum(time_array)/len(time_array))
    print()
    sys.stdout = default_sysout
    f.close()

spark.conf.set("spark.sql.autoBroadcastJoinThreshold", old_value)