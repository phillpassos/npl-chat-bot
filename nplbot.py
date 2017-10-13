#chatbot from https://chatbotsmagazine.com/contextual-chat-bots-with-tensorflow-4391749d0077


# coisas da NPL
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# coisas do Tensorflow
import numpy as np
import tflearn
import tensorflow as tf
import random


# arquivos dos intents
import json
with open('intents.json', mode='r', buffering=-1, encoding="UTF-8") as json_data:
    intents = json.load(json_data)

words = []
classes = []
documents = []
ignore_words = ['?', '!']
# loop das sentencas do arquivo e padroes
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # gerando os tokens dos padroes
        w = nltk.word_tokenize(pattern)
        # adicionando a nossa lista de palavras
        words.extend(w)
        # adicionando os docs
        documents.append((w, intent['tag']))
        # add pra lista de classes
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# filtrando as palavras
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# remocao de duplicatas
classes = sorted(list(set(classes)))

print (len(documents), "documents")
print (len(classes), "classes", classes)
print (len(words), "unique stemmed words", words)






# criando dados de treinamento
training = []
output = []
# criando array vazio pra entrada
output_empty = [0] * len(classes)

# conjunto de trainamento, pack de palavras pra cada sentenca
for doc in documents:
    # inicializa o pack de palavras
    bag = []
    # lista de palavras tokenizadas para os padroes
    pattern_words = doc[0]
    # filtrando palavras
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # array do pack de palavras
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # 0 para cada tag e '1' para a tag atual
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# misturando os recursos e passando para o np.array
random.shuffle(training)
training = np.array(training)

# criacao da lista de trainemanto e testes
train_x = list(training[:,0])
train_y = list(training[:,1])







# reset dos dados do grafico
tf.reset_default_graph()
# construcao da rede neural
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# modelo e config do tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
# iniciar treinamento (GD alg)
model.fit(train_x, train_y, n_epoch=3000, batch_size=16, show_metric=False)
model.save('model.tflearn')




# salvar estruturas de dados
import pickle
pickle.dump( {'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, open( "training_data", "wb" ) )






#----------------------- outro arquivo ----------------------------#




# restaurar estruturas
import pickle
data = pickle.load( open( "training_data", "rb" ) )
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

# importar arquivos de intent do bot
import json
with open('intents.json', mode='r', buffering=-1, encoding="UTF-8") as json_data:
    intents = json.load(json_data)





# carregar modelo salvo
model.load('./model.tflearn')








def clean_up_sentence(sentence):
    # gerando token dos padroes
    sentence_words = nltk.word_tokenize(sentence)
    # normalizando palavras
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# retornando um array do pack de palavras: 0 ou 1 pra cada palavra no pack que existe na sentenca
def bow(sentence, words, show_details=False):
    # tokens dos padroes
    sentence_words = clean_up_sentence(sentence)
    # pack de palavras
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))







# processador de respostas
# estrutura para carregar o contexto
context = {}

ERROR_THRESHOLD = 0.25
def classify(sentence):
    # gerando probabilidades do modelo
    results = model.predict([bow(sentence, words)])[0]
    # preencher predicoes abaixo do threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # ordenar por probabilidade
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # retornar a tupla do intent e probabilidade
    return return_list

def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    # se temos classificacao, entao encontramos o pack de intencoes
    if results:
        # enqunto tiver macthes, procuro os dados
        while results:
            for i in intents['intents']:
                # encontro a tag com o primeiro resultado
                if i['tag'] == results[0][0]:
                    # seta o contexto se necessario
                    if 'context_set' in i:
                        if show_details: print ('context:', i['context_set'])
                        context[userID] = i['context_set']

                    # checa se o contexto se aplica a conversacao
                    if not 'context_filter' in i or \
                        (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                        if show_details: print ('tag:', i['tag'])
                        # responsta aleatoria para o intent
                        resp = random.choice(i['responses']);
                        print(resp)
                        return resp;

            results.pop(0)


# criacao do servidor com o flask
from flask import Flask
from flask import request
import json
app = Flask(__name__, static_url_path='/static')

# criacao da rota principal (no nosso caso é a unica)
@app.route("/", methods=['POST'])
def respond():
    msg = json.loads(request.data.decode("UTF-8"));
    # respondo com o resultado do bot
    print(classify(msg["msg"]))
    msg["msg"] = response(msg["msg"])
    return json.dumps(msg), 200, {'Content-Type': 'application/json; charset=utf-8'};

# rodo o servidor
app.run(host="pat1498", port=9191)