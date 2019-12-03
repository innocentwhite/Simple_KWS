import numpy as np

def evaluate(outputs_cuda, target_raw):
    outputs = outputs_cuda.cpu()
    result_raw = []
    for i in range(len(outputs)):
        result_raw.append(np.argmax(outputs[i])) 

    classes_num = len(outputs[0])
    frr_divider = 0
    far_divider = 0
    miss = 0
    false_accept = 0
    for i in range(classes_num):
        target = (np.array(target_raw) == i)
        result = (np.array(result_raw) == i)
        xor = [a ^ b for a, b in zip(target, result)]
        miss += sum([a & b for a, b in zip(xor, target)])
        false_accept += sum([a & b for a, b in zip(xor, result)])
        frr_divider += sum(target)
        far_divider += len(target) - sum(target)
    return miss, false_accept, frr_divider, far_divider

def ROC(outputs, targets):
    outputs = outputs.cpu().detach().numpy()
    results = []
    for i in range(len(outputs)):
        results.append(np.argmax(outputs[i])) 
    FARs = []
    FRRs = []
    classes_num = len(outputs[0])
    for thrd in np.arange(1,-0.00001,-0.001).tolist():
        # macro
        inv_num = 0
        oov_num = 0
        miss = 0
        false_accept = 0
        for i in range(classes_num):
            target = (np.array(targets) == i)
            result = (np.array(outputs[:,i]) >= thrd)
            xor = [a ^ b for a, b in zip(target, result)]
            miss += sum([a & b for a, b in zip(xor, target)])
            false_accept += sum([a & b for a, b in zip(xor, result)])
            oov_num += len(target) - sum(target)
            inv_num += sum(target)
        frr = miss/inv_num
        far = false_accept/oov_num
        FARs.append(far)
        FRRs.append(frr)
    fpw = open('/home/liucl/Proj/kws-pytorch/roc.txt', 'w')
    fpw.write('FRR\tFAR\n')
    for i in range(len(FARs)):
        fpw.write('%.3f\t%.3f\n' % (FRRs[i], FARs[i]))
    fpw.close()
