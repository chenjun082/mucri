# mucri #
Codes for the IEEE TIP 2017 paper: Learning the Personalized Intransitive Preferences of Images

# Introduction  #
This repo contains the main codes for the family of MuCri models in the research paper [Learning the Personalized Intransitive Preferences of Images](http://ieeexplore.ieee.org/document/7935528/) published in [IEEE Transactions on Image Processing 2017](http://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=83).  

The MuCri models focus on learning users' personalized intransitive preferences of images. 
In many cases, we find that even if a user prefers image A to B and B to C, (s)he may still prefer C to A in pairwise comparisons.
This is mostly due to the different *criteria* that user considers in different pairwise comparisons.
Thus, this research proposes the idea of *Multi-Criterion* (short for *MuCri*) to model users' personalized intransitive preference on images.

If you are going to use the codes in this repo or discuss about this work in your research, please cite the following papers from us:
<pre>
<code>
@article{chen2017mucri,
  title={Learning the Personalized Intransitive Preferences of Images},
  author={Chen, Jun and Wang, Chaokun and Wang, Jianmin and Ying, Xiang and Wang, Xuecheng}, 
  journal={IEEE Transactions on Image Processing},
  volume={26},
  number={9},
  pages={4139 - 4153},
  year={2017},
  publisher={IEEE}
}

@inproceedings{chen2017map,
  title={Modeling the Intransitive Pairwise Image Preference from Multiple Angles},
  author={Chen, Jun and Wang, Chaokun and Wang, Jianmin},
  booktitle={ACM International Conference on Multimedia },
  year={2017},
  pages={351-359},
  publisher={ACM}
}
</code>
</pre>

# About Datasets #
The major datasets in the evaluation are [Holidays](https://github.com/chenjun082/holidays) and [Aesthetics](http://kahlan.eps.surrey.ac.uk/featurespace/fashion/) datasets, both of which contain users' pairwise comparisons on images based on their preferences.  

Especially, [Holidays](https://github.com/chenjun082/holidays) is a new dataset collected by us based on the raw images from previous [INRIA Holidays](https://link.springer.com/chapter/10.1007/978-3-540-88682-2_24) dataset. 
[Click](https://github.com/chenjun082/holidays) to see more details about the Holidays dataset.

# Code Usage #  
There are three Python source files in this repo: *dataset.py*, *evaluate.py*, *mucri.py*.

### dataset.py ###  
This file contains classes of the feed of the two datasets. 
You can load the datasets like this:
<pre>
<code>
from dataset import INRIAFeed
from dataset import AestheticFeed

feed1 = INRIAFeed()
feed1.generate_data()
    
feed2 = AestheticFeed()
feed2.generate_data()
</code>
</pre>  

<span style="color:red">Notice</span>: The codes assume that you have already put the data files in the directories *../data/inria/* and *../data/aesthetic/*.
Alternatively, you can change the directory therein.

### evaluate.py ###  
This file contains the metrics to evaluate the performance of all models. 
Currently, it provides functions to compute the *accuracy* and the *AUC* scores.
<pre>
<code>
def get_acc(x_pred, x_true) # compute accuracy score

def get_auc(x_pred, x_true) # compute auc score
</code>
</pre>

### mucri.py ###
The file contains the main codes for the MuCri models, integrated in class *MuCriPreference*.
It support five MuCri models: *L-MuCri*, *C-MuCri*, *H-MuCri-Prod*, *H-MuCri-Mean* and *H-MuCri-Max*.
You can initialize the instances of these models like this:
<pre>
<code>
model = MuCriPreference(model='latent')       # L-MuCri
model = MuCriPreference(model='content')      # C-MuCri
model = MuCriPreference(model='hybrid')       # H-MuCri-Prod
model = MuCriPreference(model='hybrid mean')  # H-MuCri-Mean
model = MuCriPreference(model='hybrid max')   # H-MuCri-Max
</code>
</pre>  

After instance initialization, you can train you model and predict on new samples.
An example to use this model is shown below:
<pre>
<code>
feed = INRIAFeed()
Xtrain, Xtest = feed.generate_data()
model = MuCriPreference(model='latent')
for train, test in zip(Xtrain, Xtest):
    model.fit(X=train,            # traning samples
              U=feed.U,           # number of users
              I=feed.I,           # number of images
              Fl=5,               # number of top-F latent criteria
              Dl=6,               # number of latent criteria
              k=100,              # dimensions of embeddings
              learning_rate=0.02, # learning rate
              lamda=.01,          # regularization parameter
              max_iters=300,      # maximum number of iterations
              tol=-1              # tolerance for early stopping
              )
    X_pred = model.predict(test, None)
    acc = get_acc(x_pred=X_pred, x_true=test)
    auc = get_auc(x_pred=X_pred, x_true=test)
    print 'Accuracy: %.5f' % acc
    print 'AUC: %.5f' % auc
</code>
</pre>

## notice ##  
To use *C-MuCri* and *H-MuCri-X*, you need to extract the content features of the images.
Since there are too many dependent libraries in our project, we just ignore them in this repo. 
Please check out the links to these libraries in our paper.
After you extract the content features, it is very simple to use them in these codes (see *mucri.py* for more detail).
