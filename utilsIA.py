import matplotlib.pyplot as plt
from sklearn.utils import resample
import pickle
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.datasets import make_blobs


sns.set_style('darkgrid', {'legend.frameon':True})

'''
============================ COMANDOS UTEIS ============================
    from keras.callbacks import EarlyStopping, ModelCheckpoint

    APAGAR VALORES NULOS - df.dropna(inplace=True) 
    DELETAR LINHA - df.drop(df.loc[df['Stock']=='Yes'].index, inplace=True)
    DELETE COLUMN - df = df.drop('column_name', 1)   / where 1 is the axis number (0 for rows and 1 for columns.)

'''


def save_object(obj, filename):
    with open(filename, 'wb') as outp: 
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def openObject(path):
    with open(f'{path}.obj', 'rb') as inp:
        tech_companies = pickle.load(inp)
        return tech_companies


class ProcessingData:

    def balancedData(df, columSample:int, max_instance=False):
        '''
            - Recebe um dataframe
            - Cria um novo df com a quantidade de cada categoria passada pelo 'max_instance'
            - Plota como ficou o df equilibrado
        '''
        df_labels = []
        f = df[columSample].astype(int).value_counts()
        n = len(f)

        for cat in f.index:
            df_labels.append(df[df[columSample]==cat])

        for sample in range(len(df_labels)):
            n_instance = max_instance if len(df_labels[sample])>max_instance else len(df_labels[sample])
            df_labels[sample] = resample(df_labels[sample], replace=True, n_samples=n_instance, random_state=0)

        df_balanced = pd.concat(df_labels) 
        GraphDatas.labelsData(df_balanced[columSample])
        return df_balanced
    
    def showLabelsQtd(df, columSample):
        tipo = type(df[columSample].values[0])
        equilibre = df[columSample].astype(tipo).value_counts()
        print(f'\nQuantidade de itens nas {len(equilibre)} categorias:\n{equilibre}')
        'Um exemplo de cada categorias'
        #one_instance_type = df.groupby(columSample, group_keys=False).apply(lambda df : df.sample(1))


    def applyFunction(df, colum_name, func):
        df[colum_name] = df[colum_name].apply(func)
        return df
    
    def encoderDf(df, columName):
        '''     le.classes_
                le.inverse_transform(array)
                le.transform(array)     '''
        le = preprocessing.LabelEncoder()
        le.fit(df[columName].values)
        return le
    


class GraphDatas:

    def labelsData(df):
        ''' 
         - Recebe a coluna de classificação do dataframe
         - plota a quantidade de cada categoria presente no DF
        '''
        df = df.value_counts()
        plt.figure(figsize=(7,7))
        my_circle = plt.Circle((0,0), 0.7, color='white')
        plt.pie(df, labels=df.index, autopct='%1.1f%%')
        p=plt.gcf()
        p.gca().add_artist(my_circle)
        plt.legend(loc='best')
        plt.title(f'Dataset - Total {df.sum()}', fontsize=30)
        plt.show()


# def evaluate_model(model, history, X_test, y_test):
#     scores = model.evaluate((X_test), y_test, verbose=0)
#     print('=========================================')
#     print("|| Accuracy: %.2f%%" % (scores[1]*100))
#     print('=========================================')

#     fig, axs = plt.subplots(1, 2, figsize=(12,6))
#     axs[0].plot(history.history['accuracy'])
#     axs[0].plot(history.history['val_accuracy'])
#     axs[0].set_title("Accuracy")
#     axs[0].legend(['Training', 'Validation'])
#     axs[1].plot(history.history['loss'])
#     axs[1].plot(history.history['val_loss'])
#     axs[1].set_title("Model- Loss")
#     axs[1].legend(['Training', 'Validation'])
#     fig.tight_layout()

# def plot_confusion_matrix(y_test, y_pred, labels, normalize=False):
#     cm = confusion_matrix(y_test, y_pred)
#     cnf_matrix = confusion_matrix(y_test, y_pred)

#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')

#     plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)#, figsize=(12,6))
#     plt.gcf().set_size_inches(17, 11)
#     plt.title('Confusion matrix')
#     plt.colorbar()
#     tick_marks = np.arange(len(labels))
#     plt.xticks(tick_marks, labels, rotation=45)
#     plt.yticks(tick_marks, labels)

#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")

#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.show()