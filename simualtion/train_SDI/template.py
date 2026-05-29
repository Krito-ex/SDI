def set_template(args):
    # Set the templates here
    if args.template.find('mst') >= 0:
        print('continue')

    if args.template.find('tsa_net') >= 0:
        print('continue')

    if args.template.find('hdnet') >= 0:
        print('continue')

    if args.template.find('mst_plus_plus') >= 0:
        args.scheduler = 'CosineAnnealingLR'

    if args.template.find('lambda_net') >= 0:
        print('continue')

    if args.template.find('restormer') >= 0:
        args.scheduler = 'CosineAnnealingLR'
        args.max_epoch = 300

    if args.template.find('fftformer') >= 0:
        args.scheduler = 'CosineAnnealingLR'
        args.max_epoch = 300

    if args.template.find('NAFNet') >= 0:
        args.scheduler = 'CosineAnnealingLR'
        args.max_epoch = 300

    if args.template.find('CSST') >= 0:
        args.scheduler = 'CosineAnnealingLR'
        args.max_epoch = 300

    if args.template.find('HSFAUT') >= 0:
        args.scheduler = 'CosineAnnealingLR'
        args.max_epoch = 300

        args.max_epoch = 300

    if args.template.find('mirnet') >= 0:
        args.scheduler = 'CosineAnnealingLR'
        args.max_epoch = 300

    if args.template.find('mprnet') >= 0:
        args.scheduler = 'CosineAnnealingLR'
        args.max_epoch = 300

    if args.template.find('unet') >= 0:
        args.scheduler = 'CosineAnnealingLR'
        args.max_epoch = 300