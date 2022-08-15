class MyNaiveClassificationModel():
    """
    Do LABEL ENCODING instead of one hot encoding
    **issue to be resolved/ to do -> original X gets changed -> need to fix

    
    LOGIC:
    It first converts all columns to categorical columns then simply equate features from test data to train data (think of it as 
    creating buckets of features -> entire n-D space divided in buckets and then choose the right bucket given our test data)
    and then calculating percent and count for that bucket.
    """
    def __init__(self, num_buckets = 20, bins_method='percentile'):
        self.num_buckets = num_buckets
        self.fit_df = None
        self.dict_bins = None
        self.dict_labels = None
        self.bins_method = bins_method
        self.dict_min_ = None
        self.dict_max_ = None
        
    def fit(self, X,y):
        num_buckets = self.num_buckets
        dict_bins = {}
        dict_labels = {}
        dict_min_ = {}
        dict_max_ = {}
        bins_method = self.bins_method
        def col_bucket(X, col, num_buckets):
            if bins_method == 'percentile':
                #idea is to do binning on percentile -> make sure equal distribution along all the buckets
                bins = sorted(list(set([np.percentile(X[col],i) for i in range(0,101,int(100/num_buckets))])))
                max_ = None
                min_ = None
            else:
                #idea is to do binnig on minmaxscaled values with fixed intervals
                try:
                    max_=np.max(X[col])
                    min_=np.min(X[col])
                    X[col] = pd.Series([((i-min_)/(max_-min_)*100) for i in list(X[col])])
                    bins = [i*(100/num_buckets) for i in range(num_buckets+1)]
                except Exception as e:
                    #this condition should not come since len(X[col].unique())<=num_buckets+1 is already there
                    print(e)
                    pass
            labels = [i for i in range(1,len(bins))]
            X[col] = pd.cut(X[col],bins=bins,labels=labels)
            #to fill the lowest value in the column since it is not included in pd.cut
            X[col].fillna(1, inplace=True)
            return X, bins, labels, max_, min_

        for col in X.columns:
            #ignoring the categorical column
            if len(X[col].unique())<=num_buckets+1:
                continue
            X, dict_bins[col], dict_labels[col], dict_max_[col], dict_min_[col] = col_bucket(X,col, num_buckets)
        # print(X.info())
        X['y'] = np.array(y)
        # print(X.info())
        self.fit_df = X.astype(float)
        self.dict_bins = dict_bins
        self.dict_labels = dict_labels
        self.dict_max_ = dict_max_
        self.dict_min_ = dict_min_
        return self
    
    def predict_proba(self, X):
        dict_bins = self.dict_bins
        dict_labels = self.dict_labels
        dict_max_ = self.dict_max_
        dict_min_ = self.dict_min_
        num_buckets = self.num_buckets
        bins_method = self.bins_method
        #y_prob -> means probability of y being == '1'
        y_prob = []
        y_count = []
        for col in X.columns:
            if col not in dict_bins.keys(): #len(X[col].unique())<=num_buckets+1:
                continue
            if bins_method == 'percentile':
                pass
            else:
                try:
                    #idea is to do maxminscaling and handle values less than min in X_test
                    X[col] = pd.Series([(((i-dict_min_[col])/(dict_max_[col]-dict_min_[col])*100) if i>dict_min_[col] else 0) for i in list(X[col])])
                except Exception as e:
                    print(e)
            X[col] = pd.cut(X[col],bins=dict_bins[col],labels=dict_labels[col])
            #to handle lowest value of the column
            X[col].fillna(1, inplace=True)
        X.reset_index(drop=True, inplace=True)
        X = X.astype(float)
        """
        fit_df = self.fit_df
        fit_df = fit_df.astype(int).astype(str)
        
        for i in range(len(X)):
            row_df = X.iloc[[i]]
            row_df = row_df.astype(int).astype(str)
            filtered_df = fit_df.merge(row_df, on = list(row_df.columns), how='inner')
            y_count.append(filtered_df['y'].count())
            try:
                prob = filtered_df['y'].sum()/filtered_df['y'].count()
            except:
                prob = np.nan
            y_prob.append(prob)
        """
        #above is less optimised approach for below
        #"""
        for i, row in X.iterrows():
            fit_df = self.fit_df
            for col in X.columns:
                fit_df = fit_df.loc[fit_df[col]==row[col]]
            y_count.append(fit_df['y'].count())
            try:
                prob = fit_df['y'].sum()/fit_df['y'].count()
            except:
                prob = np.nan
            y_prob.append(prob)
        #"""
        return y_prob #, y_count
    
    def get_params(self):
        return self.fit_df, self.dict_bins, self.dict_labels
