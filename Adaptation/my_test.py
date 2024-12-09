from my_train import *

model = torch.load('feature_extractor.bin',weights_only=False)
feature_extractor.load_state_dict(model)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold

# Hints:
# 设置 features_extractor 为 eval 模型
# 开始评估并收集特征和标签
from tqdm import tqdm
# feature_extractor.eval()
feat_list =[]
label_list = []
for i, (source_data, source_label) in enumerate(tqdm(test_dataloader)):
    source_data=source_data.cuda()
    source_label = source_label.cuda()
    feat = feature_extractor(source_data).cpu().detach().numpy()
    label_list.append(source_label.cpu().detach().numpy())
    feat_list.append(feat)

labels = np.concatenate(label_list)
labels.shape

feats = np.concatenate(feat_list)
feats.shape

# process extracted features with t-SNE
x_tsne = manifold.TSNE(n_components=2,init='random',random_state=5,verbose=1).fit_transform(feats)
# Normalization the processed features
x_min, x_max = x_tsne.min(0), x_tsne.max(0)
x_norm = (x_tsne-x_min)/(x_max-x_min)

import seaborn as sns
sne_df = np.DataFrame(x_norm,columns=['f1','f2'])
sne_df['label'] = labels
sns.scatterplot(data=sne_df,x='f1',y='f2',hue='label')
plt.show()






