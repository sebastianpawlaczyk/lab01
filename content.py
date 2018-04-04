# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 1: Regresja liniowa
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

import numpy as np
from utils import polynomial

def mean_squared_error(x, y, w):
    '''
    :param x: ciag wejsciowy Nx1
    :param y: ciag wyjsciowy Nx1
    :param w: parametry modelu (M+1)x1
    :return: blad sredniokwadratowy pomiedzy wyjsciami y
    oraz wyjsciami uzyskanymi z wielowamiu o parametrach w dla wejsc x
    '''
    return ((y - polynomial(x,w)) ** 2).mean()


def design_matrix(x_train,M):
    '''
    :param x_train: ciag treningowy Nx1
    :param M: stopien wielomianu 0,1,2,...
    :return: funkcja wylicza Design Matrix Nx(M+1) dla wielomianu rzedu M
    '''
    w = len(x_train)

    Matrix = [[x_train[y][0]**x for x in range(M+1)] for y in range(w)]

    return Matrix

def least_squares(x_train, y_train, M):
    '''
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param M: rzad wielomianu
    :return: funkcja zwraca krotke (w,err), gdzie w sa parametrami dopasowanego wielomianu, a err blad sredniokwadratowy
    dopasowania
    '''
    phi = design_matrix(x_train,M)
    phi_trans = np.transpose(phi)
    w1 = (np.dot(phi_trans,phi))
    w2 = np.linalg.inv(w1)
    w3 = np.dot(w2,phi_trans)
    w = np.dot(w3,y_train)
    return (w,mean_squared_error(x_train,y_train,w))
    pass


def regularized_least_squares(x_train, y_train, M, regularization_lambda):
    '''
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param M: rzad wielomianu
    :param regularization_lambda: parametr regularyzacji
    :return: funkcja zwraca krotke (w,err), gdzie w sa parametrami dopasowanego wielomianu zgodnie z kryterium z regularyzacja l2,
    a err blad sredniokwadratowy dopasowania
    '''
    phi = design_matrix(x_train, M)
    phi_trans = np.transpose(phi)
    w1 = np.dot(phi_trans,phi)
    idt = np.identity(len(w1))
    w2 = np.dot(regularization_lambda,idt)
    w3 = w1+w2
    w4 = np.linalg.inv(w3)
    w5 = np.dot(phi_trans,y_train)
    w6 = np.dot(w4,w5)
    return (w6,mean_squared_error(x_train,y_train,w6))
    pass


def model_selection(x_train, y_train, x_val, y_val, M_values):
    '''
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param x_val: ciag walidacyjny wejscia Nx1
    :param y_val: ciag walidacyjny wyjscia Nx1
    :param M_values: tablica stopni wielomianu, ktore maja byc sprawdzone
    :return: funkcja zwraca krotke (w,train_err,val_err), gdzie w sa parametrami modelu, ktory najlepiej generalizuje dane,
    tj. daje najmniejszy blad na ciagu walidacyjnym, train_err i val_err to bledy na sredniokwadratowe na ciagach treningowym
    i walidacyjnym
    '''
    w_m = []
    w_err = []
    for i in range(len(M_values)):
        w = least_squares(x_train,y_train,i)
        temp = w[0]
        w_m.append(temp)
        w_err.append(mean_squared_error(x_val,y_val,w_m[i]))

    s_err = w_err[0]
    index = 0
    for j in range(len(M_values)):
        if(s_err > w_err[j]):
            s_err = w_err[j]
            index = j

    return (w_m[index],mean_squared_error(x_train,y_train,w_m[index]),s_err)
    pass


def regularized_model_selection(x_train, y_train, x_val, y_val, M, lambda_values):
    '''
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param x_val: ciag walidacyjny wejscia Nx1
    :param y_val: ciag walidacyjny wyjscia Nx1
    :param M: stopien wielomianu
    :param lambda_values: lista ze wartosciami roznych parametrow regularyzacji
    :return: funkcja zwraca krotke (w,train_err,val_err,regularization_lambda), gdzie w sa parametrami modelu, ktory najlepiej generalizuje dane,
    tj. daje najmniejszy blad na ciagu walidacyjnym. Wielomian dopasowany jest wg kryterium z regularyzacja. train_err i val_err to
    bledy na sredniokwadratowe na ciagach treningowym i walidacyjnym. regularization_lambda to najlepsza wartosc parametru regularyzacji
    '''

    w_lambda = []
    w_err = []
    for i in range(len(lambda_values)):
        w = regularized_least_squares(x_train,y_train,M,lambda_values[i])
        temp = w[0]
        w_lambda.append(temp)
        w_err.append(mean_squared_error(x_val,y_val,w_lambda[i]))




    s_err = w_err[0]
    index = 0
    for j in range(len(lambda_values)):
        if (s_err > w_err[j]):
            s_err = w_err[j]
            index = j

    return (w_lambda[index],mean_squared_error(x_train,y_train,w_lambda[index]),s_err,lambda_values[index])
    pass