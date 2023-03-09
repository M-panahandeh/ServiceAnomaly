import ctypes
import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from pandas import Series
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, normalized_mutual_info_score, mean_absolute_error,r2_score,mean_absolute_percentage_error,median_absolute_error
from sklearn.svm import SVR


class MetricLearning:
    lin_ls=[]
    nonlin_ls=[]
    NMI = pd.DataFrame()
    corr=pd.DataFrame()
    def LinearRelationship_Visualize(self,metric_df,source,destination):

        self.corr = metric_df.corr()
        self.corr = np.square(self.corr)
        print(('The correlation matrix for:'+ source+ 'to' + destination), self.corr)
        # sns.heatmap(corr);
        plt.figure(figsize=(16, 6))
        heatmap = sns.heatmap(self.corr, vmin=-1, vmax=1, annot=True)
        heatmap.set_title(('Correlation Heatmap for: '+source+ 'to' + destination), fontdict={'fontsize': 12}, pad=12)
        plt.show()

    # Returns correlation matrix
    #def corrFilter(x: pd.DataFrame, bound: float):
       # xCorr = x.corr()
        #xFiltered = xCorr[((xCorr >= bound) | (xCorr <= -bound)) & (xCorr != 1.000)]
        #return xFiltered


      #remove diagonal corr and duplication
    def get_redundant_pairs(self,df):
            #Get diagonal and lower triangular pairs of correlation matrix'''
            pairs_to_drop = set()
            cols = df.columns
            for i in range(0, df.shape[1]):
                for j in range(0, i + 1):
                    pairs_to_drop.add((cols[i], cols[j]))
            #print(pairs_to_drop)
            return pairs_to_drop

    def get_top_abs_correlations(self,df,bound):
            self.lin_ls=[]
            #gets a df of metrics and gives linear regressions of metrics
            au_corr = self.corr.abs().unstack()
            labels_to_drop = self.get_redundant_pairs(df)
            # au_corr includes sorted-unrepeated correlations
            au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
            #print(au_corr)
            #high correlations
            xFiltered = au_corr[(au_corr >= bound)]
            #print(au_corr.index[0])
            #print(xFiltered)
            count=0
            if (xFiltered.size != 0):
                for i in range(0, xFiltered.size):
                    #print(xFiltered.index[2][0])
                    x=xFiltered.index[i][0]
                    y=xFiltered.index[i][1]
                    X = df[:][x].values.reshape(-1, 1)
                    Y = df[:][y]
                    regr = linear_model.LinearRegression()
                    regr.fit(X, Y)

                    Y_pred = regr.predict(X)

                    # Implement different errors
                    # R^2:
                    R_Square = r2_score(Y, Y_pred)
                    # MAE
                    mae = mean_absolute_error(Y, Y_pred)
                    # MAPE
                    mape = mean_absolute_percentage_error(Y, Y_pred)
                    # MSE
                    mse = mean_squared_error(Y, Y_pred)
                    # RMSE
                    rmse = math.sqrt(mean_squared_error(Y, Y_pred))
                    # MedAE
                    medae = median_absolute_error(Y, Y_pred)
                    # Maximum error
                    max_error = max(abs(Y - Y_pred))
                    num_data = X.shape[0]

                    # mse = mean_squared_error(Y, Y_pred)
                    # #rmse = math.sqrt(mse / num_data)
                    # num_data= X.shape[0]
                    # rse = math.sqrt(mse / (num_data - 2))
                    # #rsquare = linear_regressor.score(X, Y)
                    # #mae = mean_absolute_error(Y, Y_pred)
                    # mae=max(abs(Y- Y_pred))

                    #print(str(Y- Y_pred))
                    #print(str(max(Y- Y_pred)))

                    #draw Y-Y_pred for each metric
                    # p = Y-Y_pred
                    # q = np.array(range(0, len(p)))
                    # plt.plot(q, p)

                    print('There is a linear correlation between of: '+y+'='+str(regr.coef_)+ '*'+x+'+'+ str(regr.intercept_)+'-+'+str(rmse))
                    print('\n-------------------------------------------------------------\n')
                    #save result in a dictionary
                    self.lin_ls.append({'metric1':y,'metric2':x,'coef':(regr.coef_), 'intercept':(regr.intercept_),'error':rmse,'reg':regr})
                    #print the number of relarions per edge
                   # count=count+1
            #ctypes.windll.user32.MessageBoxW(0,"number of linear relations per edge" + str(count),"", 1)

                   # print('Coefficients: \n', regr.coef_)


    def NonLinearRelationship_Visualize(self,metric_df,source, destination):
        # create NMI matrix
        NMI_matrix = np.zeros((6, 6))
        for i in range(0, 6):
            for j in range(0, 6):
                NMI_matrix[i][j] = normalized_mutual_info_score(metric_df.iloc[:, i], metric_df.iloc[:, j])
        NMI_matrix
        #heatmap
        plt.figure(figsize=(16, 6))
        heatmap = sns.heatmap(NMI_matrix, vmin=-1, vmax=1, annot=True)
        heatmap.set_title(('NMI_matrix Heatmap for: ' + source + 'to' + destination), fontdict={'fontsize': 12},
                              pad=12)
        plt.show()
        #convet numpy.array of NMI to df
        self.NMI=pd.DataFrame(NMI_matrix, columns = ['request_duration', 'request_byte', 'response_byte', 'queue_size',
                    'latency', 'throughput'],index=['request_duration', 'request_byte', 'response_byte', 'queue_size',
                    'latency', 'throughput'])



    def get_non_linear(self, df, NMI_bound,corr_bound):
            count=0
            self.nonlin_ls=[]
            # gets a df of metrics and gives NMI cof of metrics
            au_NMI = self.NMI.abs().unstack()
            labels_to_drop = self.get_redundant_pairs(df)
            # au_corr includes sorted-unrepeated correlations
            au_NMI = au_NMI.drop(labels=labels_to_drop).sort_values(ascending=False)
            # high NMI
            xFiltered_high_NMI = au_NMI[(au_NMI >= NMI_bound)]

            #get low corr
            # gets a df of metrics and gives linear regressions of metrics
            au_corr = self.corr.abs().unstack()
            labels_to_drop = self.get_redundant_pairs(df)
            # au_corr includes sorted-unrepeated correlations
            au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
            # low correlations
            xFiltered_low_correlation = au_corr[(au_corr < corr_bound)]

            #intersection of high NMI and low corr for two colums that are metric names
            #print((xFiltered_high_NMI.index[:][0:][0:]))
            NMI_metrics = np.intersect1d(xFiltered_high_NMI.index[:][0:][0:],xFiltered_low_correlation.index[:][0:][0:])
            NMI_metrics= pd.Series(NMI_metrics)

            #print(type(NMI_metrics.index[0][0]))
            if (NMI_metrics.size!=0):
                for i in range(0, NMI_metrics.size):
                      #print(NMI_metrics.index[0][1])
                     x = NMI_metrics[i][0]
                     y = NMI_metrics[i][1]
                     X = df[:][x].values.reshape(-1, 1)
                     Y = df[:][y]
                     #print(X)
                     #print(Y)
                     # Fit regression model : C:hyperparameter
                     svr_rbf = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
                     #svr_lin = SVR(kernel="linear", C=100, gamma="auto")
                     #svr_poly = SVR(kernel="poly", C=100, gamma="auto", degree=3, epsilon=0.1, coef0=1)
                     svr=svr_rbf.fit(X, Y)
                    #yfit = svr.predict(x)
                    # print("MSE:", mean_squared_error(y, yfit))

                      #error
                     y_predict = svr.predict(X)
                     #mse = mean_squared_error(Y, y_predict)
                     num_data = X.shape[0]

                     # Implement different errors
                     # R^2:
                     R_Square = r2_score(Y, y_predict)
                     # MAE
                     mae = mean_absolute_error(Y, y_predict)
                     # MAPE
                     mape = mean_absolute_percentage_error(Y, y_predict)
                     # MSE
                     mse = mean_squared_error(Y, y_predict)
                     # RMSE
                     rmse = math.sqrt(mean_squared_error(Y, y_predict))
                     # MedAE
                     medae = median_absolute_error(Y, y_predict)
                     # Maximum error
                     max_error = max(abs(Y - y_predict))

                     #print("ERORRS: "+"mae: "+mae+ " mape: "+mape+ " mse: "+mse+" rmse: "+rmse+" medae: "+medae+" max: "+max_error)
                     #rmse = math.sqrt(mse / num_data)
                     #mae = mean_absolute_error(Y, y_predict)
                     #mae = max(abs(Y - y_predict))

                     # score = svr.score(x, y)
                     # print("R-squared:", score)
                    # rse = math.sqrt(mse / (num_data - 2))


                     print('There is a non-linear relashionship between of: '+x+' and '+y+' :'+'y_i*alpha_i='+str(svr.dual_coef_)
                            +' +Intercept='+str(svr.intercept_)+'-+' + str(rmse))
                     print('\n-------------------------------------------------------------\n')
                     # save result in a dictionary
                     self.nonlin_ls.append({'metric1': y, 'metric2': x,'svr':svr, 'coef': (svr.dual_coef_), 'intercept': (svr.intercept_),
                                    'error': rmse})

                    # count = count + 1
            #ctypes.windll.user32.MessageBoxW(0, "number of non-linear relations per edge" + str(count),"", 1)

                # # print('Coefficients: \n', regr.coef_)



                # # Look at the results of different SVR
                # lw = 2
                #
                # svrs = [svr_rbf, svr_lin, svr_poly]
                # kernel_label = ["RBF", "Linear", "Polynomial"]
                # model_color = ["m", "c", "g"]
                #
                # fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10), sharey=True)
                # for ix, svr in enumerate(svrs):
                #     axes[ix].plot(
                #         X,
                #         svr.fit(X, y).predict(X),
                #         color=model_color[ix],
                #         lw=lw,
                #         label="{} model".format(kernel_label[ix]),
                #     )
                #     axes[ix].scatter(
                #         X[svr.support_],
                #         y[svr.support_],
                #         facecolor="none",
                #         edgecolor=model_color[ix],
                #         s=50,
                #         label="{} support vectors".format(kernel_label[ix]),
                #     )
                #     axes[ix].scatter(
                #         X[np.setdiff1d(np.arange(len(X)), svr.support_)],
                #         y[np.setdiff1d(np.arange(len(X)), svr.support_)],
                #         facecolor="none",
                #         edgecolor="k",
                #         s=50,
                #         label="other training data",
                #     )
                #     axes[ix].legend(
                #         loc="upper center",
                #         bbox_to_anchor=(0.5, 1.1),
                #         ncol=1,
                #         fancybox=True,
                #         shadow=True,
                #     )
                #
                # fig.text(0.5, 0.04, "data", ha="center", va="center")
                # fig.text(0.06, 0.5, "target", ha="center", va="center", rotation="vertical")
                # fig.suptitle("Support Vector Regression", fontsize=14)
                # plt.show()
