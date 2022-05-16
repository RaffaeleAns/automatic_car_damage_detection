import os
import pickle
import argparse

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("py.debug")
logger.addFilter(lambda record: "PIL.Image:Error" not in record.getMessage())



parser = argparse.ArgumentParser(description='TL_6')
parser.add_argument('--min_budget',   type=float, help='Minimum number of epochs for training.',    default=5)
parser.add_argument('--max_budget',   type=float, help='Maximum number of epochs for training.',    default=15)
parser.add_argument('--n_iterations', type=int,   help='Number of iterations performed by the optimizer', default=25)
parser.add_argument('--worker', help='Flag to turn this into a worker process', action='store_true')
parser.add_argument('--run_id', type=str, help='A unique run id for this optimization run. An easy option is to use the job id of the clusters scheduler.', default='TL_6')
parser.add_argument('--nic_name',type=str, help='Which network interface to use for communication.', default=  'lo' )
parser.add_argument('--shared_directory',type=str, help='A directory that is accessible for all processes, e.g. a NFS share.', default='.')
parser.add_argument('--backend',help='Toggles which worker is used. Choose between a pytorch and a keras implementation.', choices=['pytorch', 'keras'], default='keras')

args=parser.parse_args()


from worker import KerasWorker as worker

# Every process has to lookup the hostname
host = hpns.nic_name_to_host(args.nic_name)


if args.worker:
    import time
    time.sleep(5)   # short artificial delay to make sure the nameserver is already running
    w = worker(rootDir='masked_images_split', run_id=args.run_id, host=host, timeout=120)
    w.load_nameserver_credentials(working_directory=args.shared_directory)
    w.run(background=False)
    exit(0)


result_logger = hpres.json_result_logger(directory=args.shared_directory, overwrite=True)


# Start a nameserver:
NS = hpns.NameServer(run_id=args.run_id, host=host, port=0, working_directory=args.shared_directory)
#NS.shutdown()
ns_host, ns_port = NS.start()

# Start local worker
w = worker(rootDir='masked_images_split', run_id=args.run_id, host=host, nameserver=ns_host, nameserver_port=ns_port, timeout=120)
w.run(background=True)

# Run an optimizer
bohb = BOHB(  configspace = worker.get_configspace(),
                      run_id = args.run_id,
                      host=host,
                      nameserver=ns_host,
                      nameserver_port=ns_port,
                      result_logger=result_logger,
                      min_budget=args.min_budget, max_budget=args.max_budget,
               )
res = bohb.run(n_iterations=args.n_iterations)

# store results
with open(os.path.join(args.shared_directory, 'results.pkl'), 'wb') as fh:
    pickle.dump(res, fh)

# shutdown
bohb.shutdown(shutdown_workers=True)
NS.shutdown()
