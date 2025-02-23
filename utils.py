import numpy as np
import pickle, json


def read_file(filename):
    with open(filename) as f:
        lines = f.read().splitlines()
    return lines


def pickle_load(path):
    with open(path, 'rb') as fid:
        data_ = pickle.load(fid)
    return data_


def pickle_save(path, data):
    with open(path, 'wb') as fid:
        pickle.dump(data, fid)


def json_load(path):
    with open(path, 'r') as fid:
        data_ = json.load(fid)
    return data_


def json_save(path, data):
    with open(path, 'w') as fid:
        json.dump(data, fid, indent=4, sort_keys=True)


def compute_ap(ranks, nres):
    """
    Computes average precision for given ranked indexes.
    
    Arguments
    ---------
    ranks : zerro-based ranks of positive images
    nres  : number of positive images
    
    Returns
    -------
    ap    : average precision
    """

    # number of images ranked by the system
    nimgranks = len(ranks)

    # accumulate trapezoids in PR-plot
    ap = 0

    recall_step = 1. / nres

    for j in np.arange(nimgranks):
        rank = ranks[j]

        if rank == 0:
            precision_0 = 1.
        else:
            precision_0 = float(j) / rank

        precision_1 = float(j + 1) / (rank + 1)

        ap += (precision_0 + precision_1) * recall_step / 2.

    return ap


def compute_map(ranks, gnd, kappas=[]):
    """
    Computes the mAP for a given set of returned results.

         Usage: 
           map = compute_map (ranks, gnd) 
                 computes mean average precsion (map) only
        
           map, aps, pr, prs = compute_map (ranks, gnd, kappas) 
                 computes mean average precision (map), average precision (aps) for each query
                 computes mean precision at kappas (pr), precision at kappas (prs) for each query
        
         Notes:
         1) ranks starts from 0, ranks.shape = db_size X #queries
         2) The junk results (e.g., the query itself) should be declared in the gnd stuct array
         3) If there are no positive images for some query, that query is excluded from the evaluation
    """

    map = 0.
    nq = len(gnd) # number of queries
    aps = np.zeros(nq)
    pr = np.zeros(len(kappas))
    prs = np.zeros((nq, len(kappas)))
    nempty = 0

    for i in np.arange(nq):
        qgnd = np.array(gnd[i]['ok'])

        # no positive images, skip from the average
        if qgnd.shape[0] == 0:
            aps[i] = float('nan')
            prs[i, :] = float('nan')
            nempty += 1
            continue

        try:
            qgndj = np.array(gnd[i]['junk'])
        except:
            qgndj = np.empty(0)

        ###########################################################
        # pp = np.in1d(ranks[:100,i], qgnd).tolist()
        # nn = [not xx for xx in pp]
        # foo = np.concatenate([np.arange(100)[np.array(pp)], np.arange(100)[np.array(nn)]])
        # bar = ranks[:, i][foo]
        # ranks[:100, i] = bar
        ###########################################################

        # sorted positions of positive and junk images (0 based)
        pos  = np.arange(ranks.shape[0])[np.in1d(ranks[:,i], qgnd)]
        junk = np.arange(ranks.shape[0])[np.in1d(ranks[:,i], qgndj)]

        k = 0
        ij = 0
        if len(junk):
            # decrease positions of positives based on the number of
            # junk images appearing before them
            ip = 0
            while (ip < len(pos)):
                while (ij < len(junk) and pos[ip] > junk[ij]):
                    k += 1
                    ij += 1
                pos[ip] = pos[ip] - k
                ip += 1

        # compute ap
        ap = compute_ap(pos, len(qgnd))
        map = map + ap
        aps[i] = ap

        # compute precision @ k
        pos += 1 # get it to 1-based
        for j in np.arange(len(kappas)):
            kq = min(max(pos), kappas[j]); 
            prs[i, j] = (pos <= kq).sum() / kq
        pr = pr + prs[i, :]

    map = map / (nq - nempty)
    pr = pr / (nq - nempty)

    return map, aps, pr, prs


def compute_metrics(dataset, ranks, gnd, kappas=[1, 5, 10]):
    
    # old evaluation protocol
    if dataset.startswith('classic'):
        map, aps, _, _ = compute_map(ranks, gnd)
        out = {'map': np.around(map*100, decimals=3)}
        print('>> {}: mAP {:.2f}'.format(dataset, out['map']))

    # new evaluation protocol
    elif dataset.startswith('revisited'):
        
        gnd_t = []
        for i in range(len(gnd)):
            g = {}
            g['ok'] = np.concatenate([gnd[i]['easy']])
            g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['hard']])
            gnd_t.append(g)
        mapE, apsE, mprE, prsE = compute_map(ranks, gnd_t, kappas)

        gnd_t = []
        for i in range(len(gnd)):
            g = {}
            g['ok'] = np.concatenate([gnd[i]['easy'], gnd[i]['hard']])
            g['junk'] = np.concatenate([gnd[i]['junk']])
            gnd_t.append(g)
        mapM, apsM, mprM, prsM = compute_map(ranks, gnd_t, kappas)

        gnd_t = []
        for i in range(len(gnd)):
            g = {}
            g['ok'] = np.concatenate([gnd[i]['hard']])
            g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['easy']])
            gnd_t.append(g)
        mapH, apsH, mprH, prsH = compute_map(ranks, gnd_t, kappas)


        out = {
            'E_map': np.around(mapE*100, decimals=3),
            'M_map': np.around(mapM*100, decimals=3),
            'H_map': np.around(mapH*100, decimals=3),
            'E_mp':  np.around(mprE*100, decimals=3),
            'M_mp':  np.around(mprM*100, decimals=3),
            'H_mp':  np.around(mprH*100, decimals=3),
            # 'apsE': apsE.tolist(),
            # 'apsM': apsM.tolist(),
            # 'apsH': apsH.tolist(),
        }

        # with open('medium.txt', 'w') as f:
        #     f.write('\n'.join([str(v) for v in apsM]))
        # with open('hard.txt', 'w') as f:
        #     f.write('\n'.join([str(v) for v in apsH]))
        # with open('easy.txt', 'w') as f:
        #     f.write('\n'.join([str(v) for v in apsE]))

        print('>> {}: mAP E: {}, M: {}, H: {}'.format(dataset, out['E_map'], out['M_map'], out['H_map']))
        print('>> {}: mP@k{} E: {}, M: {}, H: {}'.format(dataset, kappas, out['E_mp'], out['M_mp'], out['H_mp']))

    return out