### This notebook contains the functions needed for the sensor sensitivity models creation from the sensors datasets ###
# Written by- Rehan Mendis (14/07/2026)


######################## Import packages ######################## 

import argparse
import random
import warnings
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from firthlogist import FirthLogisticRegression

from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages

from scipy.stats import gaussian_kde

from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, ConfusionMatrixDisplay,  confusion_matrix, 
f1_score, log_loss, precision_recall_curve, precision_score, recall_score, roc_auc_score,roc_curve)
from sklearn.utils import resample

from statsmodels.nonparametric.smoothers_lowess import lowess

warnings.filterwarnings("ignore")



######################## Functions for the sensor sensitivity model ######################## 

# 1. Function to get a Bootstrap sample
def Bootstrap_sample(dataset, seed = 1234):
    """
    Perform a two-stage hierarchical bootstrap.

    Parameters
    ----------
    dataset :Input dataset containing leak locations and sensor observations.
    seed    : Random seed for reproducibility.

    Returns
    -------
    X_trF    : Standardised training features.
    y_tr     : Training labels.
    X_testF  : Standardised test features.
    y_test   : Test labels.
    std_valF : Feature standard deviations used for scaling.
    """
    
    # Unique leak values and bootstrap train test samples
    leak_values = list(dataset['leak_dat'].unique()) 
    train = resample(leak_values , n_samples = len(leak_values), random_state = seed)
    test = np.array([x for x in leak_values if x not in train]) 
    sample_count = [[x,train.count(x)] for x in set(train)]
    
    # Obtaining bootstrap dataset-1 (with leak values)
    dataset1 = pd.DataFrame()
    dataset_len = 0
    for category, count in sample_count:
        index = 1
        while (index <= count):
            dat = dataset[(dataset['leak_dat'] == category)]
            dataset_len += len(dat)
            dataset1 = pd.concat([dataset1, dat], sort=False, ignore_index=True)
            index +=1
    test_dataset =  dataset[dataset["leak_dat"].isin(list(test))].reset_index(drop = True)
    
    # Obtaining bootstrap dataset-2
    seed_new = seed + 1
    dataset2 = resample(dataset1 , n_samples = len(dataset1), random_state = seed_new).reset_index(drop = True)
    X_tr = dataset2.drop(columns=['alert', 'leak_dat'])
    y_tr = dataset2[['alert']]
    std_valF = X_tr.std()
    std_valF[std_valF == 0] = 1
    X_trF = X_tr / std_valF
    X_test = test_dataset.drop(columns=['alert', 'leak_dat'])
    y_test = test_dataset[['alert']]
    X_testF = X_test / std_valF
    
    return [ X_trF, y_tr, X_testF, y_test, std_valF]


# 2. Function to get a Bootstrap coefficients and results on each sample
def Bootstrap_coefs(dataset, iterations = 3000, intersecpt_val = True, seedF = 1234, firth=False):
    """
    Estimate logistic regression coefficients using hierarchical bootstrap sampling.

    For each bootstrap iteration, leak locations are sampled with replacement and used to generate a bootstrap dataset. 
    A logistic regression model is then fitted and evaluated.

    Parameters
    ----------
    dataset :Training dataset containing leak size, distance-profile features, leak identifiers and binary sensor alerts.
    iterations :   Number of bootstrap replications. (default=3000)
    intersecpt_val :   Whether an intercept term should be estimated.(default=True)
    seedF : Random seed for reproducible bootstrap sampling. (default=1234)
    firth : If True, fit a Firth penalised logistic regression model.If False, fit a standard logistic regression model.
        (default=False)

    Returns
    -------
    If iterations == 1
        Returns model coefficients, performance metrics and diagnostic quantities for a single bootstrap model.

    If iterations > 1
        Returns bootstrap coefficient distributions, summary performance metrics, 
        metric distributions and the average confusion matrix.

    Notes
    -----
    Predictor variables are standardised using the standard deviation of the bootstrap training sample.

    De-standardised coefficients are obtained by dividing the fitted coefficients by the corresponding
    feature standard deviations.

    The hierarchical bootstrap reflects the structure of the dataset,
    where multiple sensor observations are associated with each leak location.
    
    When firth=True, coefficient estimation is performed using Firth penalised likelihood, which can improve estimation
    stability in small samples and in the presence of separation.
    """
    
    # Select model type
    if firth:
        logistic_model = FirthLogisticRegression( fit_intercept=intersecpt_val,max_iter=4000,skip_ci=True,skip_pvals=True)
    else:
        logistic_model = LogisticRegression(random_state=0, penalty=None, fit_intercept=intersecpt_val,
                                            max_iter=4000)
        
    print(f"Model Type: {'Firth Logistic Regression' if firth else 'Logistic Regression'}")

    
    if iterations == 1:
        # =====================================================
        # Model Training
        # =====================================================
        X_tr, y_tr, X_test, y_test, std_vals,  = Bootstrap_sample(dataset, seed = seedF)
        logistic_model.fit(X_tr,y_tr)
        predict_proba_tr = logistic_model.predict_proba(X_tr)[:, 1]
        
        y_pred=logistic_model.predict(X_test)
        y_predprob=logistic_model.predict_proba(X_test)

        accuracy=accuracy_score(y_test,y_pred)*100
        rocauc = roc_auc_score(y_test, y_predprob[:,1])
        lg_log_loss = log_loss(y_test, y_predprob)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        ap = average_precision_score(y_test, y_predprob[:,1])
        

        print('Y Predictions ',y_pred)
        print('Accuracy Rate ',accuracy)
        print('ROC AUC ',rocauc)
        print('log loss ',lg_log_loss)
        print('Precision ',precision)
        print('Recall ',recall)
        print('F1 ',f1)
        print('Average Precision (AP) ',ap)

        coefficients=logistic_model.coef_
        if coefficients.ndim == 2:
            coefficients = coefficients[0]
        coefs_denorm  = coefficients /std_vals.values
        intercept = np.asarray(logistic_model.intercept_)
        if intercept.size == 1:
            intercept = intercept.item()
        
        print('Normalised Coefficients ',coefficients,'\nIntercept ',intercept)
        print('De-Normalised Coefficients ',coefs_denorm)
        
        with PdfPages("ModelDiagnostics.pdf") as pdf:
        
            # =====================================================
            # Model Performance Summary
            # =====================================================
            fig1, axes = plt.subplots(2, 2, figsize=(14, 12))

            # Plot Confusion Matrix
            disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test,y_pred), 
                                          display_labels=['No Alert', 'Alert'])
            disp.plot(ax=axes[0,0])
            for labels in disp.text_.ravel():
                labels.set_fontsize('x-large')
            axes[0,0].tick_params(axis='both', labelsize='x-large')
            axes[0,0].set_xlabel('Predicted label', fontsize='x-large')
            axes[0,0].set_ylabel('True label', fontsize='x-large')
            axes[0,0].set_title('(a) Confusion Matrix', fontsize='x-large')
            axes[0, 0].text(0.6, 0.03,  
                    f'Precision = {precision:.2f}\n' f'Recall = {recall:.2f}\n' f'F1 = {f1:.2f}',
                            transform=axes[0, 0].transAxes, fontsize='x-large',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=1.0))

            # Plot ROC Curve
            fpr, tpr, thresholds = roc_curve(y_test, y_predprob[:,1])

            axes[0,1].plot(fpr, tpr, label=f'ROC Curve (AUC = {rocauc:.2f})')
            axes[0,1].plot([0, 1], [0, 1], 'r--', label='Random Classifier')
            axes[0,1].set_xlabel('False Positive Rate', fontsize='x-large')
            axes[0,1].set_ylabel('True Positive Rate', fontsize='x-large')
            axes[0,1].set_title('(b) Receiver Operating Characteristic (ROC) Curve', fontsize='x-large')
            axes[0,1].legend(loc='lower right', fontsize='x-large')
            axes[0,1].grid(True)

            # Plot PR-Curve
            precision_pr_auc, recall_pr_auc, pr_thresholds = precision_recall_curve(y_test, y_predprob[:, 1])

            baseline = y_test.iloc[:, 0].mean()
            axes[1,0].plot(recall_pr_auc,precision_pr_auc,label=f'PR Curve (AP = {ap:.2f})',color='darkorange')
            axes[1,0].axhline( y=baseline,  color='red',linestyle='--',label=f'Baseline = {baseline:.2f})')
            axes[1,0].set_xlabel('Recall', fontsize='x-large')
            axes[1,0].set_ylabel('Precision', fontsize='x-large')
            axes[1,0].set_title('(c) Precision-Recall Curve', fontsize='x-large')
            axes[1,0].legend(loc='upper right', fontsize='x-large')
            axes[1,0].grid(True)

            # Calibration Curve
            fraction_of_positives, mean_predicted_value = calibration_curve( y_test,y_predprob[:, 1], n_bins=5,
                                                                            strategy='quantile')


            axes[1, 1].plot( [0, 1],[0, 1], 'k--',label='Perfect Calibration')
            axes[1, 1].plot( mean_predicted_value, fraction_of_positives,   's-', linewidth=2.5, markersize=8,
                            label='Logistic Regression')
            axes[1, 1].set_xlabel( 'Mean Predicted Probability', fontsize='x-large')
            axes[1, 1].set_ylabel('Fraction of Positives',fontsize='x-large')
            axes[1, 1].set_title( '(d) Calibration Curve',fontsize='x-large')
            bins = pd.qcut(y_predprob[:,1], q=5, duplicates='drop')
            axes[1, 1].legend(fontsize='large')
            axes[1, 1].grid(True)


            plt.tight_layout() 
            pdf.savefig(fig1, bbox_inches='tight')
            plt.show()
            plt.close(fig1)



            # =====================================================
            # Deviance residual diagnostics
            # =====================================================

            p = np.asarray(predict_proba_tr)
            y_tr_diag = np.asarray(y_tr).ravel()
            p = np.clip(p, 1e-15, 1 - 1e-15)

            continuous_vars = ['leak_size','dist_metals','dist_non_metal','dist_metal_comp']

            renamed_vars = ['Leak Size','Metals','Non Metal','Metal Composite' ]

            # Deviance residuals
            resid = np.where(y_tr_diag == 1, np.sqrt(-2 * np.log(p)),  -np.sqrt(-2 * np.log(1 - p)))

            fig2, axes = plt.subplots(2, 2, figsize=(12, 8))
            axes = axes.flatten()
            for ax, var, varsN in zip(axes, continuous_vars, renamed_vars):
                x = X_tr[var]
                ax.scatter(x, resid, alpha=0.5)
                smoothed = lowess(  resid, x, frac=0.3)
                ax.plot(smoothed[:, 0],  smoothed[:, 1], color='red', linewidth=2)
                ax.axhline(0, color='black', linestyle='--')
                ax.set_xlabel(varsN,fontsize='x-large')
                ax.set_ylabel('Deviance Residual', fontsize='x-large')
                ax.set_title(f'Deviance Residuals vs {varsN}',fontsize='x-large')

            plt.tight_layout()
            pdf.savefig(fig2, bbox_inches='tight')
            plt.show()
            plt.close(fig1)

        # =====================================================
        # Return model results
        # =====================================================
        
        return [[coefficients, coefs_denorm, intercept], accuracy, rocauc, lg_log_loss, precision, recall, f1,
               [predict_proba_tr, X_tr,y_tr]]

    
    
    coefs_all                    = []
    coefs_denorm_all             = []
    interscept_all               = []
    accuracy_tr_all              = []
    rocauc_tr_all                = []
    lg_log_loss_tr_all           = []
    precision_tr_all             = []
    recall_tr_all                = []
    f1_tr_all                    = []
    average_precision_tr_all     = []
    accuracy_tst_all             = []
    rocauc_tst_all               = []
    lg_log_loss_tst_all          = []
    precision_tst_all            = []
    recall_tst_all               = []
    f1_tst_all                   = []
    average_precision_tst_all    = []
    confusion_mat_tst_all = []
    
    random.seed(seedF)
    # ====================================================+
    # Model Training
    # =====================================================
    for index in range(iterations):
        print(f'----- Iteration {index+1} -----')

        seedF2 = random.randint(1, 10000)

        X_tr, y_tr, X_test, y_test, std_vals = Bootstrap_sample(dataset, seed=seedF2)

        # Skip bootstrap samples containing only one class
        if (y_tr['alert'].nunique() < 2) or (y_test['alert'].nunique() < 2):
            print(f'Iteration {index+1}: skipped (single class in training or  data)' )
            continue

        logistic_model.fit(X_tr, y_tr)

        # Train
        y_pred_tr     =logistic_model.predict(X_tr)
        y_predprob_tr =logistic_model.predict_proba(X_tr)

        accuracy_tr=accuracy_score(y_tr,y_pred_tr)*100
        accuracy_tr_all.append(accuracy_tr)
        rocauc_tr = roc_auc_score(y_tr, y_predprob_tr[:,1])
        rocauc_tr_all.append(rocauc_tr)
        lg_log_loss_tr = log_loss(y_tr, y_predprob_tr)
        lg_log_loss_tr_all.append(lg_log_loss_tr)
        precision_tr = precision_score(y_tr, y_pred_tr)
        precision_tr_all.append(precision_tr)
        recall_tr = recall_score(y_tr, y_pred_tr)
        recall_tr_all.append(recall_tr)
        f1_tr = f1_score(y_tr, y_pred_tr)
        f1_tr_all.append(f1_tr)
        average_precision_tr = average_precision_score(y_tr, y_predprob_tr[:,1])
        average_precision_tr_all.append(average_precision_tr)

        print('---- Training Info ----')
        print('Accuracy Rate ',accuracy_tr,end=" \ ")
        print('ROC AUC ',rocauc_tr,end=" \ ")
        print('log loss ',lg_log_loss_tr,end=" \ ")
        print('Precision ', precision_tr, end=" \ ")
        print('Recall ', recall_tr, end=" \ ")
        print('F1 ', f1_tr, end=" \\")
        print('Average Precision (AP) ',average_precision_tr)


        # Test
        y_pred=logistic_model.predict(X_test)
        y_predprob=logistic_model.predict_proba(X_test)

        accuracy=accuracy_score(y_test,y_pred)*100
        accuracy_tst_all.append(accuracy)
        rocauc = roc_auc_score(y_test, y_predprob[:,1])
        rocauc_tst_all.append(rocauc)
        lg_log_loss = log_loss(y_test, y_predprob)
        lg_log_loss_tst_all.append(lg_log_loss)
        precision_tst = precision_score(y_test, y_pred)
        precision_tst_all.append(precision_tst)
        recall_tst = recall_score(y_test, y_pred)
        recall_tst_all.append(recall_tst)
        f1_tst = f1_score(y_test, y_pred)
        f1_tst_all.append(f1_tst)
        precision_pr_tst, recall_pr_tst, _ = precision_recall_curve(y_test, y_predprob[:,1])
        average_precision_tst = average_precision_score(y_test, y_predprob[:,1])
        average_precision_tst_all.append(average_precision_tst)
        confusion_mat_tst = confusion_matrix(y_test, y_pred)
        confusion_mat_tst_all.append(confusion_mat_tst)

        print('---- Test Info ----')
        #print('Y Predictions ',y_pred,end=" \ ")
        print('Accuracy Rate ',accuracy,end=" \ ")
        print('ROC AUC ',rocauc,end=" \ ")
        print('log loss ',lg_log_loss,end=" \ ")
        print('Precision ', precision_tst, end=" \ ")
        print('Recall ', recall_tst, end=" \ ")
        print('F1 ', f1_tst, end=" \ ")
        print('Average Precision (AP) ',average_precision_tst)
        
        coefficients = np.asarray(logistic_model.coef_).squeeze()
        coefs_all.append(coefficients)
        coefs_denorm = coefficients / std_vals.values
        coefs_denorm_all.append(coefs_denorm)
        intercept = logistic_model.intercept_
        interscept_all.append(intercept)

        print('Coefficients ',coefficients,'& Intercept ',intercept)
        print('----- Iteration end  -----')
    
    accuracy_tr_mean      = ( sum(accuracy_tr_all) / len(accuracy_tr_all) )
    accuracy_test_mean    = ( sum(accuracy_tst_all) / len(accuracy_tst_all) )
    rocauc_tr_mean        =  ( sum(rocauc_tr_all) / len(rocauc_tr_all) )
    rocauc_test_mean      =  ( sum(rocauc_tst_all) / len(rocauc_tst_all) )
    logloss_tr_mean       =  ( sum(lg_log_loss_tr_all) / len(lg_log_loss_tr_all) )
    logloss_test_mean     =  ( sum(lg_log_loss_tst_all) / len(lg_log_loss_tst_all) )
    precision_tr_mean     = sum(precision_tr_all) / len(precision_tr_all)
    precision_test_mean   = sum(precision_tst_all) / len(precision_tst_all)
    recall_tr_mean        = sum(recall_tr_all) / len(recall_tr_all)
    recall_test_mean      = sum(recall_tst_all) / len(recall_tst_all)
    f1_tr_mean            = sum(f1_tr_all) / len(f1_tr_all)
    f1_test_mean          = sum(f1_tst_all) / len(f1_tst_all)
    ap_tr_mean            = sum(average_precision_tr_all) / len(average_precision_tr_all)
    ap_test_mean          = sum(average_precision_tst_all) / len(average_precision_tst_all)
    avg_confusion_mat_tst = np.round(np.mean(confusion_mat_tst_all, axis=0), 0)
    
    
    # =====================================================
    # Return model results
    # =====================================================
    return ([coefs_all, coefs_denorm_all, interscept_all], 
            [accuracy_tr_mean, accuracy_test_mean,  rocauc_tr_mean, rocauc_test_mean, logloss_tr_mean,  logloss_test_mean,
             precision_tr_mean,  precision_test_mean,    recall_tr_mean,  recall_test_mean,   
             f1_tr_mean,    f1_test_mean,  ap_tr_mean,   ap_test_mean],
            [ accuracy_tr_all,  rocauc_tr_all,  lg_log_loss_tr_all,precision_tr_all, recall_tr_all,   f1_tr_all,   
             average_precision_tr_all   ], 
            [accuracy_tst_all,  rocauc_tst_all,  lg_log_loss_tst_all,  precision_tst_all,  recall_tst_all, f1_tst_all,
             average_precision_tst_all], avg_confusion_mat_tst)

    
########################################################################################  
    
    
    
######################## Functions for visualisations  ######################## 

# 1. Function visual results with error bars
def errorbar_plotF(material1, material1_CI, material1_col = 'blue', label1 = 'Metal',
                   material2 = None, material2_CI = None, material2_col =  'red', label2 = 'Metal Comp',
                   material3 = None, material3_CI = None, material3_col = 'green', label3 = 'Non Metal', 
                   rangeF= (-3, 0), hori_line = 0.2,  savefig=False, filename='errorbar_plot.pdf'):
    """
    Plot bootstrap coefficient distributions and confidence intervals.

    Parameters
    ----------
    material1 : Bootstrap coefficient estimates for the first material category.
    material1_CI : Confidence interval in the format [lower, median, upper].
    material1_col : Histogram colour for the first material category. (default='blue')
    label1 :  Legend label for the first material category. (default='Metal')
    material2 :Bootstrap coefficient estimates for the second material category (default=None)
    material2_CI :Confidence interval for the second material category (default=None)
    material2_col :   Histogram colour for the second material category (default='red')
    label2 :  Legend label for the second material category  (default='Metal Comp')
    material3 :   Bootstrap coefficient estimates for the third material category (default=None)
    material3_CI :  Confidence interval for the third material category (default=None)
    material3_col :  Histogram colour for the third material category (default='green')
    label3 :   Legend label for the third material category  (default='Non Metal')
    rangeF :  Histogram x-axis range. (default=(-3, 0))
    hori_line :  Vertical position used for plotting confidence intervals. (default=0.2)
    savefig : Save the figure to file. (default=False)
    filename : Output filename used when savefig=True.


    Returns
    ------- 
    None
        Displays the histogram plot.
    """
  
    
    hori_line_minus = hori_line - (hori_line*0.3)
    hori_line_plus  = hori_line + (hori_line*0.3)
    
    # plotting histogram with one material type
    plt.hist(material1, label= label1, density=True,  range=rangeF, alpha=0.5,  color= material1_col, bins = 50)
    plt.hlines(y=hori_line, xmin=material1_CI[0], xmax=material1_CI[2], colors='black', linestyles='dashed', 
               linewidth=2, label= label1 + ' Confidence Interval')
    plt.vlines(x=material1_CI[0], ymin=hori_line_minus, ymax=hori_line_plus, colors='black',
               linewidth=2)
    plt.vlines(x=material1_CI[2], ymin=hori_line_minus, ymax=hori_line_plus, colors='black',
               linewidth=2)

    # plotting histogram with two material type
    if material2 is not None:
        plt.hist(material2, label=label2, density=True, range=rangeF, alpha=.5, color=material2_col, bins = 50)
        plt.hlines(y=hori_line, xmin=material2_CI[0], xmax=material2_CI[2], colors='deeppink', linestyles='dashed', 
                   linewidth=2, label= label2 + ' Confidence Interval')
        plt.vlines(x=material2_CI[0], ymin=hori_line_minus, ymax=hori_line_plus, colors='deeppink',
                   linewidth=2)
        plt.vlines(x=material2_CI[2], ymin=hori_line_minus, ymax=hori_line_plus, colors='deeppink',
                   linewidth=2)

         
    # plotting histogram with three material type
    if material3 is not None:
        plt.hist(material3, label=label3, density=True, range=rangeF, alpha=.5,  color=material3_col, bins = 50)
        plt.hlines(y=hori_line, xmin=material3_CI[0], xmax=material3_CI[2], colors='lime', linestyles='dashed', 
                   linewidth=2, label= label3 + ' Confidence Interval')
        plt.vlines(x=material3_CI[0], ymin=hori_line_minus, ymax=hori_line_plus, colors='lime',
                   linewidth=2)
        plt.vlines(x=material3_CI[2], ymin=hori_line_minus, ymax=hori_line_plus, colors='lime',
                   linewidth=2)


    plt.legend(loc='upper left')
    if savefig:
        plt.savefig(filename, bbox_inches='tight', dpi=600)

    plt.show()    
    
    
    
# 2. Joint Distribution Plot
def JointDistPlot(InihibtionArray1, InihibtionArray2, InihibtionArray3, label1, label2, label3,
                  level1 = 0.7, level2 = 0.5, level3 = 0.3, rangeF12 = [(-3,0), (-3,0)],
                 rangeF13 = [(-3,0), (-3,0)], rangeF23 = [(-3,0), (-3,0)], savefig=True, filename='JointDistPlot.pdf'):
    """
    Plot pairwise joint distributions of bootstrap coefficient estimates.

    The function creates three pairwise joint-distribution plots using
    2D histograms and Gaussian Kernel Density Estimation (KDE). Confidence
    sets are displayed as contour lines corresponding to the specified
    probability levels.

    Parameters
    ----------
    InihibtionArray1 : Bootstrap coefficient estimates for the first material category.
    InihibtionArray2 :  Bootstrap coefficient estimates for the second material category.
    InihibtionArray3 :  Bootstrap coefficient estimates for the third material category.
    label1 :  Label for the first material category.
    label2 : Label for the second material category.
    label3 :   Label for the third material category.
    level1 :  Largest confidence-set probability level. (default=0.7)
    level2 : Intermediate confidence-set probability level. (default=0.5)
    level3 : Smallest confidence-set probability level. (default=0.3)
    rangeF12 :  Plotting ranges for the first-second material comparison in the format [(xmin, xmax), (ymin, ymax)].
          (default=[(-3, 0), (-3, 0)])
    rangeF13 :    Plotting ranges for the first-third material comparison in the  format [(xmin, xmax), (ymin, ymax)].
          (default=[(-3, 0), (-3, 0)])
    rangeF23 :  Plotting ranges for the second-third material comparison in the format [(xmin, xmax), (ymin, ymax)].
        (default=[(-3, 0), (-3, 0)])
    savefig : Save the figure to file. (default=False)
    filename : Output filename used when savefig=True.

    Returns
    -------
    None
        Displays three pairwise joint-distribution plots with KDE-based
        confidence sets.

    Notes
    -----
    The confidence sets are derived from Gaussian kernel density estimates.
    A dashed diagonal line (x = y) is included in each subplot to facilitate comparison between coefficient estimates.
    A shared colour scale is used across all subplots to ensure consistent interpretation of density levels.
    """
    
    pairs = [   (InihibtionArray1, InihibtionArray2, label1, label2, rangeF12 ),
                (InihibtionArray1, InihibtionArray3, label1, label3, rangeF13),
                (InihibtionArray2, InihibtionArray3, label2, label3, rangeF23)   ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    
    # Function to find density threshold for a given confidence level
    def level(prob):
        return z_sorted[np.searchsorted(cumulative, prob)]
    
    # Compute global min/max for shared normalization ---
    all_counts = []

    for X, Y, _, _ , ranges in pairs:
        rangeFX, rangeFY = ranges # unpack the two ranges
        counts, _, _ = np.histogram2d(X, Y, bins=50, range=[rangeFX, rangeFY])
        all_counts.append(counts)

    global_min = min(c.min() for c in all_counts)
    global_max = max(c.max() for c in all_counts)
    norm = Normalize(vmin=global_min, vmax=global_max)

    
    for i, (ax, (X, Y, labX, labY, ranges)) in enumerate(zip(range(3), pairs)):
        
        ax = axes[i]
        rangeFX, rangeFY = ranges # unpack the two ranges

        # Create 2D histogram
        counts, xedges, yedges, img = ax.hist2d(X, Y, bins=50, cmap="plasma", range=[rangeFX, rangeFY],norm=norm)
        #counts -The bi-dimensional histogram values x and y. Values in x are histogrammed along the first dimension
                                          # and values in y are histogrammed along the second dimension.
        # xedges - The bin edges along the x-axis.
        # yedges - The bin edges along the y-axis.


        # Compute KDE on a grid for smooth contours ( Obtaining the mid points for meshgrid from histogram edges )
        xx, yy = np.meshgrid(  0.5 * (xedges[:-1] + xedges[1:]),
                               0.5 * (yedges[:-1] + yedges[1:])  )   

        kde = gaussian_kde([X, Y])                                       # KDE learning
        # Applying the meshgrid and reshape to original dims
        zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

        # Convert density to cumulative probability for confidence levels
        z_sorted = np.sort(zz.ravel())[::-1]    # Sort in descending order
        cumulative = np.cumsum(z_sorted)
        cumulative /= cumulative[-1]

        # Confidence levels
        levels = sorted([level(level1), level(level2), level(level3)])

        # Visualisation
        cols = ["lightpink", "violet", "crimson"]
        ax.contour(xx, yy, zz, levels=levels, colors=cols)


        # --- Legend only in first subplot ---
        if i == 0:
            labels = [f"{int(l*100)}%" for l in [level1, level2, level3]]
            handles = [Line2D([0], [0], color=c, lw=3, label=lab)for c, lab in zip(cols, labels)]
            ax.legend(handles=handles, title="Confidence Sets",facecolor='white', loc='upper left')
            
            # --- Shared colorbar ---
            cax = ax.inset_axes([-0.12, 0.05, 0.03, 0.9])   # adjust as needed
            cbar = plt.colorbar(img, cax=cax, shrink=0.85, pad=0.02)
            cax.yaxis.tick_left()
            cax.yaxis.set_label_position('left')

        # --- Move axes ---
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
        
        
        # --- Add x = y diagonal ---
        xmin, xmax = rangeFX
        ymin, ymax = rangeFY
        low = min(xmin, ymin)
        high = max(xmax, ymax)
        ax.plot([low, high], [low, high], color='lightblue', linestyle='--', linewidth=2)
        ax.set_xlabel(labX, fontsize=15, fontweight='heavy', fontfamily= 'serif')
        ax.set_ylabel(labY, fontsize=15, fontweight='heavy', fontfamily= 'serif')
        

    plt.tight_layout()
    if savefig:
        plt.savefig(filename, bbox_inches='tight', dpi=600)

    plt.show()   
        
if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser(description="Sensor sensitivity model using hierarchical bootstrap logistic regression.")
    parser.add_argument( "--dataset", required=True, help="Path to input dataset (.csv)")
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--firth", action="store_true")
    parser.add_argument("--savefig", action="store_true", help="Save visualisation figures.")
    args = parser.parse_args()
    
    dataset = pd.read_csv(args.dataset)
    results = Bootstrap_coefs( dataset=dataset, iterations=args.iterations,  seedF=args.seed,  firth=args.firth )
    
    if args.iterations >1:
        coefs = np.vstack(results[0][1])

        metals_coef      = coefs[:,1]
        non_metal_coef   = coefs[:,2]
        metal_comp_coef  = coefs[:,3]

        metal_CI = np.percentile( metals_coef, [2.5, 50, 97.5])
        non_metal_CI = np.percentile(  non_metal_coef,  [2.5, 50, 97.5])
        metal_comp_CI = np.percentile(   metal_comp_coef,  [2.5, 50, 97.5])

        errorbar_plotF(  metals_coef,  metal_CI,  material2=metal_comp_coef,  material2_CI=metal_comp_CI,
            material3=non_metal_coef, material3_CI=non_metal_CI, savefig=args.savefig)
        
        JointDistPlot( metals_coef, metal_comp_coef, non_metal_coef, 'Metal', 'Metal Composite', 'Non Metal',
                      savefig=args.savefig)
        

    print(f"***** Done in {time.time() - start}s *****")
