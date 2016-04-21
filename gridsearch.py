def get_results(n_components,max_features,criterion,n_estimators):
    kf = KFold(n=len(X), n_folds=12, shuffle=True, random_state=21)
    d = []
    for train_index, test_index in kf:
        for rf_features,criter,rf_estimators in itertools.product(max_features,criterion,n_estimators):
            X_tr, X_test = X[train_index], X[test_index]
            X_train, X_validation = train_test_split(X_tr, test_size=0.2, random_state=21)
            Y_tr, Y_test = Y[train_index], Y[test_index]
            Y_train, Y_validation = train_test_split(Y_tr, test_size=0.2, random_state=21)
            del X_tr
            del Y_tr
            forest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion=criter,
                    max_depth=None, max_features=rf_features, max_leaf_nodes=None,
                    min_samples_leaf=1, min_samples_split=2,
                    min_weight_fraction_leaf=0.0, n_estimators=rf_estimators, n_jobs=1,
                    oob_score=False, random_state=21, verbose=0,
                    warm_start=False)
            forest.fit(X_train,Y_train)
            forest_pred = forest.predict(X_validation)
            name = '%s,%s,%s'%(str(rf_features),str(criter),str(rf_estimators))
            d.append((name,accuracy_score(Y_validation,forest_pred),X_train,X_test,Y_train,Y_test))
            #forest_probs = forest.predict_proba(X_test)
            #sig_clf = CalibratedClassifierCV(forest, method="sigmoid", cv="prefit")
            #sig_clf.fit(X_validation, Y_validation)
            #sig_clf_probs = sig_clf.predict_proba(X_test)
            #sig_score = log_loss(Y_test, sig_clf_probs)
    return sorted(d, key=lambda tup: tup[1])[-1]
