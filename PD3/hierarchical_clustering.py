import numpy as np
from scipy.spatial.distance import cdist

def hierarchical_clustering(X, n_clusters):
    n = X.shape[0]

    if type(X) != np.ndarray:  # obsługa wyjątków
        raise ValueError("Argument X musi być macierzą typu numpy.ndarray.")
    if n_clusters <= 0:
        raise ValueError("Liczba skupień n_clusters musi być większa od 0.")
    if n_clusters > n:
        raise ValueError("n_clusters nie może być większe niż liczba punktów w X.")
    if n == 0:
        return np.array([])

    C = [[i] for i in range(n)]   # zakładamy, że każde skupienie zawiera jeden element
    D = cdist(X, X, metric="euclidean")   # liczona jest macierz odległości między punktami
    np.fill_diagonal(D, np.inf)   # na diagonali są odległości punktów od nich samych; dajemy nieskończoność abby potem funkcja nie brała tych odległości pod uwagę

    active_clusters = n   # aktualna liczba skupień

    while active_clusters > n_clusters:   # w pętli schodzimy z n skupień do wymaganej wejściowej liczby skupień
        dist_min = np.argmin(D)   # w macierzy odległości znajdujemy najmniejszą wartość
        i, j = divmod(dist_min, D.shape[0])   # znajdujemy punkty, między którymi jest właśnie ta odległość najmniejsza

        C[i].extend(C[j])   # łączone są skupienia Ci i Cj w jedno skupienie
        C[j] = []   # usuwamy osobne skupienie Cj
        
        new_D = np.minimum(D[i, :], D[j, :])   # aktualizacja macierzy odległości
        D[i, :] = new_D
        D[:, i] = new_D
        D[i, i] = np.inf
        D[j, :] = np.inf
        D[:, j] = np.inf
        
        active_clusters -= 1   # od liczby skupień, którą mieliśmy do tej pory usuwamy jednno skupienie Cj

    labels = np.zeros(n, dtype=int)   # tworzymy pusty wektor na etykiety skupień dla punktów
    final_clusters = [c for c in C if len(c)>0]   # mamy wektor list punktów przyporządkowanych do odpowiednich skupień
    for cluster_id, points in enumerate(final_clusters):   # uzupełniamy wektor etykietami, czyli oznaczeniami skupień, do których należą poszczególne punkty
        for p in points:
            labels[p] = cluster_id
            
    return labels   # funkcja wywołuje wektor etykiet, czyli numer skupienia do, kórego należy każdy z punktów (ideksów w wektorze)