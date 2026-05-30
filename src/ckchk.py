import torch
for tag in ['sas_d256_best', 'sas_d512_best']:
    try:
        sd = torch.load(f'out/{tag}.pt', map_location='cpu')
    except Exception as e:
        print(tag, 'LOAD FAIL', e); continue
    tot = sum(int(torch.isnan(v).sum()) for v in sd.values())
    print(tag, 'total NaN params:', tot)
    iw = sd['item.weight']
    print('  item.weight mean', round(float(torch.nan_to_num(iw).mean()), 4), 'nan_rows', int(torch.isnan(iw).any(1).sum()), '/', iw.shape[0])
    for k, v in sd.items():
        n = int(torch.isnan(v).sum())
        if n:
            print('  NaN in', k, n, '/', v.numel())
