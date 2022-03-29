from math import pi
import matplotlib.pyplot as plt
import pandas as pd
# import stanza 
# stanza.download('en')
# stanza_nlp = stanza.Pipeline('en')

def polar_plot(df):
    '''Polar plot a dataframe grouped accross a single dimension'''
    # number of variable
    categories=list(df)[0:]
    N = len(categories)
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    # Initialise the spider plot
    plt.figure(figsize=(10,10))
    ax = plt.subplot(111, polar=True) 
    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories)
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([1,2,3,4,5], ["1","2","3","4","5"], color="grey", size=7)
    plt.ylim(0,5)
    # Plot 
    for d in df.index:
        values=df.loc[d].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=d)
        ax.fill(angles, values, 'b', alpha=0.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1),frameon=False,title='Aggregated over '+df.index.name)
    # plt.title('Gdoor aggregated by '+df.index.name)
    plt.show()

def to_table(anno):
    word_list, word_len, word_upos = [], [], []
    for sentence in anno.sentences:
        for word in sentence.words:
            word_list.append(word.lemma)
            word_len.append(len(word.text))
            word_upos.append(word.upos)

    word_df = {'word'   : word_list, 
            'length' : word_len,     
            'pos'    : word_upos
            }
    word_df = pd.DataFrame(word_df)

    noun_df = word_df[(word_df['pos'] == 'NOUN') &  (word_df['length'] > 1)]
    
    return noun_df
    