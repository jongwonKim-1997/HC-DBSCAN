def import_data(data='mnist', size=1000):
    # MNIST

    if data == "mnist":
        mnist_train = dsets.MNIST(root='MNIST_data/', # 다운로드 경로 지정
                                train=True, # True를 지정하면 훈련 데이터로 다운로드
                                transform=transforms.ToTensor(), # 텐서로 변환
                                download=True)

      
        batch_size = 60000
        data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                drop_last=True)
                            
        for X, Y in data_loader: # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y느 ㄴ레이블.
            # image is already size of (28x28), no reshape
            # label is not one-hot encoded
            X = X.to(device).numpy()
            Y = Y.to(device).numpy()
        train_data = X.reshape(len(X),-1)
        train_labels = Y 
    # REUTERS   
    if data =='reuters':
        train_data = pd.read_csv("./data/20_newsgroup.csv")
        train_data = train_data.dropna()
        train_data['labels'] = pd.Categorical(train_data.copy()['labels']).codes
        train_labels = train_data['labels']
        train_data = train_data.drop(['labels'],axis=1)

    if data =="cifar10":
        mnist_train = dsets.CIFAR10(root='MNIST_data/', # 다운로드 경로 지정
                                train=True, # True를 지정하면 훈련 데이터로 다운로드
                                transform=transforms.ToTensor(), # 텐서로 변환
                                download=True)

        batch_size = 50000
        data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                drop_last=True)
                            
        for X, Y in data_loader: # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y느 ㄴ레이블.
            # image is already size of (28x28), no reshape
            # label is not one-hot encoded
            X = X.to(device).numpy()
            Y = Y.to(device).numpy()
        train_data = X.reshape(len(X),-1)
        train_labels = Y 

    if data =="FashionMNIST":
        mnist_train = dsets.FashionMNIST(root='FashionMNIST/', # 다운로드 경로 지정
                                train=True, # True를 지정하면 훈련 데이터로 다운로드
                                transform=transforms.ToTensor(), # 텐서로 변환
                                download=True)

        batch_size = 60000
        data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                drop_last=True)
                            
        for X, Y in data_loader: # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y느 ㄴ레이블.
            # image is already size of (28x28), no reshape
            # label is not one-hot encoded
            X = X.to(device).numpy()
            Y = Y.to(device).numpy()
        train_data = X.reshape(len(X),-1)
        train_labels = Y 

    if data =='iris':
        train_data = pd.read_csv("./data/iris.csv")
        train_data = train_data.dropna()
        train_labels = train_data['iris']
        train_data['iris'] = pd.Categorical(train_data.copy()['iris']).codes
        train_labels = train_data['iris']
        train_data = train_data.drop(['iris'],axis=1)
    if data =='Cell237':
        train_data = pd.read_csv("./data/Cell237.csv")
        train_data = train_data.dropna()
        train_labels = train_data['class']
        train_data['class'] = pd.Categorical(train_data.copy()['class']).codes
        train_labels = train_data['class']
        train_data = train_data.drop(['class'],axis=1)


    if data =='seeds':
        train_data = pd.read_csv("./data/seeds.csv")
        train_data = train_data.dropna()
        train_labels = train_data['seeds']
        train_data = train_data.drop(['seeds'],axis=1)

    if data =='table_1':
        train_data = pd.read_csv("./data/abalone.csv")
        train_data = train_data.dropna()
        train_labels = train_data['Rings']
        train_data = train_data.drop(['Rings'],axis=1)

    if data =='table_2':
        train_data = pd.read_csv("./data/Data_Cortex_Nuclear.csv")
        train_data = train_data.dropna()
        train_labels = train_data['class']
        train_data['class'] = pd.Categorical(train_data.copy()['class']).codes
        train_labels = train_data['class']
        train_data = train_data.drop(['class'],axis=1)

    if data =='table_3':
        train_data = pd.read_csv("./data/Dry_Bean_Dataset.csv")
        train_data = train_data.dropna()
        train_labels = train_data['Class']
        train_data['Class'] = pd.Categorical(train_data.copy()['Class']).codes
        train_labels = train_data['Class']
        train_data = train_data.drop(['Class'],axis=1)

    if data =='table_4':
        train_data = pd.read_csv("./data/Faults.csv")
        train_data = train_data.dropna()
        train_labels = train_data['class']
        train_data['class'] = pd.Categorical(train_data.copy()['class']).codes
        train_labels = train_data['class']
        train_data =  train_data.drop(['class'],axis=1)

    if data =='table_5':    
        train_data = pd.read_csv("./data/Frogs_MFCCs.csv")
        train_data = train_data.dropna()
        train_labels = train_data['Species']
        train_data['Species'] = pd.Categorical(train_data.copy()['Species']).codes
        train_labels = train_data['Species']
        train_data = train_data.drop(['Species','Family','Genus'],axis=1)

    if data == 'toy1':

        im = imageio.imread('test_2.png')
        #im = imageio.imread('test_smile_face.png')
        im_data = np.sum(im,axis=2)
        im_data_list = []
        for i in range(400):
            for j in range(400):
                if im_data[i,j] !=765:
                    im_data_list.append([i,j])
        train_data = np.array(im_data_list)
        train_labels = np.array([0]*(len(train_data)-1)+[1])
    if data == 'toy2':

        im = imageio.imread('test_smile_face.png')
        im_data = np.sum(im,axis=2)
        im_data_list = []
        for i in range(400):
            for j in range(400):
                if im_data[i,j] !=765:
                    im_data_list.append([i,j])
        train_data = np.array(im_data_list)
        train_labels = np.array([0]*(len(train_data)-1)+[1])
    if data == 'toy3':
        train_data = pd.read_csv("./data/Compound.csv")
        train_data = train_data.dropna()
        train_labels = np.array([0]*(len(train_data)-1)+[1])
    if data == 'toy4':
        train_data = pd.read_csv("./data/pathbased.csv")
        train_data = train_data.dropna()
        train_labels = np.array([0]*(len(train_data)-1)+[1])
    if data == 'toy5':
        train_data = pd.read_csv("./data/Aggregation.csv")
        train_data = train_data.dropna()
        train_labels = np.array([0]*(len(train_data)-1)+[1])
    if data == 'toy6':
        train_data = pd.read_csv("./data/spiral.csv")
        train_data = train_data.dropna()
        train_labels = np.array([0]*(len(train_data)-1)+[1])


        

   #print(train_data.shape)
    size = min(size,len(train_data))
    rnd_idx = np.random.RandomState(seed=2022).permutation(len(train_data))[:size]
    raw_data = train_data
    raw_data = np.array(raw_data)
    train_data = raw_data[rnd_idx]
    raw_labels =np.array(train_labels)
    
   #print(raw_labels.shape)

    train_labels = raw_labels[rnd_idx]
    
    train_labels = train_labels.reshape(-1)

    return train_data, train_labels
def embedding_data(train_data ,embedding = 'umap',n_components=2):


    # UMAP
    if embedding == 'umap':
        train_data = umap.UMAP(n_components = n_components,n_neighbors=10, min_dist=0.001).fit_transform(train_data)

    # PCA 
    if embedding == 'pca':
        pca = PCA(n_components=n_components)
        train_data = pca.fit_transform(train_data)

    return train_data

