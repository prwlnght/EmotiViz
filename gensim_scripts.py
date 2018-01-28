import gensim, logging, os

isconverting = False #flag for doing some converting work
ismodeling = False#flag for building models. WARNING: does not check if models already exist
istestingmodels = True


if(ismodeling): #gave up on the idea of saving and reusing models, pickling c like models doens't seem to work
    vocab_directory = '/Users/hayden/workspace/EmotViz/vocabularies/word2vec/'
    model_dump_directory = '/Users/hayden/workspace/EmotViz/my_models/'
    for filename in os.listdir(vocab_directory):
        vocab_file = os.path.join(vocab_directory, filename)
        m_file_name = ''.join(filename.split('.')[:-1])
        model_file_name = os.path.join(model_dump_directory, m_file_name)
        model = gensim.models.KeyedVectors.load_word2vec_format(
            vocab_file,
            binary=False)
        model.save(model_file_name)
        print(model_file_name)



if(istestingmodels): #gave up on the idea of saving and reusing models, pickling c like models doens't seem to work
    model_dir = '/Users/hayden/workspace/EmotViz/models/'
    model_name = 'w2vglovetwitter27B100d'
    models = gensim.models.Word2Vec.load(os.path.join(model_dir, model_name))
    #model = gensim.models.Word2Vec(os.path.join(model_dir, model_name), min_count=1)
    print(model.similarity('woman', 'man'))
    print('Success')













if(isconverting):
    base_dir = '/Users/hayden/workspace/EmotViz/vocabularies/'
    input_dir = 'glove/'
    output_dir = 'word2vec/'
    for filename in os.listdir(base_dir+input_dir):
        if filename.endswith(".txt"):
            input_file = os.path.join(base_dir+input_dir, filename)
            output_file = os.path.join(base_dir+output_dir, 'w2v.'+filename)
            print(output_file)
            gensim.scripts.glove2word2vec.glove2word2vec(
                input_file, output_file)


