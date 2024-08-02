def update_ssl_backbone(tgr_weight, src_weight, prefix):
    print(f"updating backbone")
    updated_count = 0
    unupdated_count = 0
    for key in tgr_weight.state_dict():
        query = prefix + key
        if query in src_weight:
            updated_count += 1
            tgr_weight.state_dict()[key].copy_(src_weight[query])
        else:
            print(f'{query} not in src model')
            unupdated_count += 1
    print(f'{updated_count=} {unupdated_count=}')