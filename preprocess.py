import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import PorterStemmer

# 停用詞去除（Stopword Removal）: 如 a/ an、 the 、 is/ are等
def remove_stopwords(text: str) -> str:
    '''
    E.g.,
        text: 'Here is a dog.'
        preprocessed_text: 'Here dog.'
    '''
    stop_word_list = stopwords.words('english')
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(text) 
    tokens = [token.strip() for token in tokens]  # 去除首尾空格
    filtered_tokens = [token for token in tokens if token.lower() not in stop_word_list]
    preprocessed_text = ' '.join(filtered_tokens)
    # print('after remove_stopward: ', preprocessed_text)
    return preprocessed_text


def preprocessing_function(text: str) -> str:
    preprocessed_text = remove_stopwords(text)
    
    # TO-DO 0: Other preprocessing function attemption
    # Begin your code 

    # ## method 1: Lowercase Conversion
    # # preprocessed_text = preprocessed_text.lower()
    # preprocessed_text = text.lower()

    # ## method 2: Stemming
    # tokenizer = ToktokTokenizer()
    # tokens = tokenizer.tokenize(preprocessed_text)
    # port = PorterStemmer()
    # stemmed_port = [port.stem(token) for token in tokens]
    # preprocessed_text = ' '.join(stemmed_port)

    # ## method 3: remove <br / >, 句號
    # # 句號可能不能移喔
    # preprocessed_text = preprocessed_text.replace('<br / >', ' ')
    # preprocessed_text = preprocessed_text.replace('&amp', '')

    # ## method 4: remove '....'
    # preprocessed_text = ' '.join([w for w in preprocessed_text.split() if '..' not in w] )
    
    # ## method 5: remove single letter
    # preprocessed_text = ' '.join([w for w in preprocessed_text.split() if len(w)>1])



    # print('after preprocessing_function: ', preprocessed_text) 
    # End your code

    return preprocessed_text