# ============================================================
# From KamiPan Lab
# ============================================================
# For readability, we slightly refined the code structure of the scripts after the experiments were completed. 
# As a result, the released code may differ slightly from the exact version used in the paper, but the core functionality remains unchanged and can still serve as a useful reference. 
# The entry is on the bottom of this script.
# If you have any questions, please contact the corresponding author.

# The attack names 'DBA-style' and 'BBA' are legacy names retained from the early stage of this research, corresponding to 'BBA-GS' and 'BBA-RS' in the paper, respectively.

# Enjoy:)

import os
import gc

CPU_COUNT = os.cpu_count() or 1
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
N_JOBS = max(1, CPU_COUNT - 1)
TORCH_NUM_THREADS = int(os.environ.get("TORCH_NUM_THREADS", "1"))
TORCH_INTEROP_THREADS = int(os.environ.get("TORCH_INTEROP_THREADS", "1"))

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot
from xgboost import plot_importance
from lightgbm import plot_importance
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold,StratifiedKFold
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import itertools
from joblib import Parallel, delayed
# Reduce thread conflicts on macOS
import torch
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")
import torch.nn as nn
import torch.optim as optim

without_time_list = ['average_time_between_incoming_transactions_normal', 'average_time_between_outcoming_transactions_normal',
                    'time_since_the_first_until_the_last_transaction_normal', 'longest_interval_between_two_transactions_normal',
                    'shortest_interval_between_two_transactions_normal', 'total_number_of_transactions_normal',
                    'the_number_of_unique_outcoming_addresses_normal', 'the_number_of_unique_incoming_addresses_normal',
                    'the_total_number_of_incoming_transactions_normal', 'the_total_number_of_outcoming_transactions_normal',
                    'the_proportion_of_unique_outcoming_address_transactions_normal','the_proportion_of_unique_incoming_address_transactions_normal',
                    'proportion_of_outcoming_address_transactions_normal', 'proportion_of_incoming_address_transactions_normal',
                    'minimum_value_in_Ether_ever_received_normal', 'maximum_value_in_Ether_ever_received_normal',
                    'avg_value_in_Ether_ever_received_normal', 'minimum_value_in_Ether_ever_sent_normal',
                    'maximum_value_in_Ether_ever_sent_normal', 'avg_value_in_Ether_ever_sent_normal',
                    'total_value_in_Ether_ever_received_normal', 'total_value_in_Ether_ever_sent_normal',
                    'the_number_of_transactions_per_day_normal', 'the_number_of_incoming_transactions_per_day_normal',
                    'the_number_of_outcoming_transactions_per_day_normal', 'the_number_of_incoming_transactions_per_hour_normal',
                    'the_number_of_outcoming_transactions_per_hour_normal', 'the_number_of_incoming_amounts_per_day_normal',
                    'the_number_of_outcoming_amounts_per_day_normal', 'the_total_number_of_amounts_outcoming_plus_incoming_normal',
                    'the_number_of_incoming_amounts_per_hour_normal', 'the_number_of_outcoming_amounts_per_hour_normal',
                    'the_number_of_transactions_per_hour_normal', 'the_number_of_amounts_per_day_normal',
                    'the_number_of_amounts_per_hour_normal', 'reverted_numbers_normal',
                    'average_time_between_incoming_transactions_internal', 'average_time_between_outcoming_transactions_internal',
                    'time_since_the_first_until_the_last_transaction_internal', 'longest_interval_between_two_transactions_internal',
                    'shortest_interval_between_two_transactions_internal', 'total_number_of_transactions_internal',
                    'the_number_of_unique_outcoming_addresses_internal', 'the_number_of_unique_incoming_addresses_internal',
                    'the_total_number_of_incoming_transactions_internal', 'the_total_number_of_outcoming_transactions_internal',
                    'the_proportion_of_unique_outcoming_address_transactions_internal','the_proportion_of_unique_incoming_address_transactions_internal',
                    'proportion_of_outcoming_address_transactions_internal', 'proportion_of_incoming_address_transactions_internal',
                    'minimum_value_in_Ether_ever_received_internal', 'maximum_value_in_Ether_ever_received_internal',
                    'avg_value_in_Ether_ever_received_internal', 'minimum_value_in_Ether_ever_sent_internal',
                    'maximum_value_in_Ether_ever_sent_internal', 'avg_value_in_Ether_ever_sent_internal',
                    'total_value_in_Ether_ever_received_internal', 'total_value_in_Ether_ever_sent_internal',
                    'the_number_of_transactions_per_day_internal', 'the_number_of_incoming_transactions_per_day_internal',
                    'the_number_of_outcoming_transactions_per_day_internal', 'the_number_of_incoming_transactions_per_hour_internal',
                    'the_number_of_outcoming_transactions_per_hour_internal', 'the_number_of_incoming_amounts_per_day_internal',
                    'the_number_of_outcoming_amounts_per_day_internal', 'the_total_number_of_amounts_outcoming_plus_incoming_internal',
                    'the_number_of_incoming_amounts_per_hour_internal', 'the_number_of_outcoming_amounts_per_hour_internal',
                    'the_number_of_transactions_per_hour_internal', 'the_number_of_amounts_per_day_internal',
                    'the_number_of_amounts_per_hour_internal', 'reverted_numbers_internal',
                    'average_time_between_incoming_transactions_erc20', 'average_time_between_outcoming_transactions_erc20',
                    'time_since_the_first_until_the_last_transaction_erc20', 'longest_interval_between_two_transactions_erc20',
                    'shortest_interval_between_two_transactions_erc20', 'total_number_of_transactions_erc20',
                    'the_number_of_unique_outcoming_addresses_erc20', 'the_number_of_unique_incoming_addresses_erc20',
                    'the_total_number_of_incoming_transactions_erc20', 'the_total_number_of_outcoming_transactions_erc20',
                    'the_proportion_of_unique_outcoming_address_transactions_erc20','the_proportion_of_unique_incoming_address_transactions_erc20',
                    'proportion_of_outcoming_address_transactions_erc20', 'proportion_of_incoming_address_transactions_erc20',
                    'minimum_value_in_Ether_ever_received_erc20', 'maximum_value_in_Ether_ever_received_erc20',
                    'avg_value_in_Ether_ever_received_erc20', 'minimum_value_in_Ether_ever_sent_erc20',
                    'maximum_value_in_Ether_ever_sent_erc20', 'avg_value_in_Ether_ever_sent_erc20',
                    'total_value_in_Ether_ever_received_erc20', 'total_value_in_Ether_ever_sent_erc20',
                    'the_number_of_transactions_per_day_erc20', 'the_number_of_incoming_transactions_per_day_erc20',
                    'the_number_of_outcoming_transactions_per_day_erc20', 'the_number_of_incoming_transactions_per_hour_erc20',
                    'the_number_of_outcoming_transactions_per_hour_erc20', 'the_number_of_incoming_amounts_per_day_erc20',
                    'the_number_of_outcoming_amounts_per_day_erc20', 'the_total_number_of_amounts_outcoming_plus_incoming_erc20',
                    'the_number_of_incoming_amounts_per_hour_erc20', 'the_number_of_outcoming_amounts_per_hour_erc20',
                    'the_number_of_transactions_per_hour_erc20', 'the_number_of_amounts_per_day_erc20',
                    'the_number_of_amounts_per_hour_erc20',
                    'average_time_between_incoming_transactions_nft', 'average_time_between_outcoming_transactions_nft',
                    'time_since_the_first_until_the_last_transaction_nft', 'longest_interval_between_two_transactions_nft',
                    'shortest_interval_between_two_transactions_nft', 'total_number_of_transactions_nft',
                    'the_number_of_unique_outcoming_addresses_nft', 'the_number_of_unique_incoming_addresses_nft',
                    'the_total_number_of_incoming_transactions_nft', 'the_total_number_of_outcoming_transactions_nft',
                    'the_proportion_of_unique_outcoming_address_transactions_nft', 'the_proportion_of_unique_incoming_address_transactions_nft',
                    'proportion_of_outcoming_address_transactions_nft', 'proportion_of_incoming_address_transactions_nft',
                    'minimum_value_in_Ether_ever_received_nft', 'maximum_value_in_Ether_ever_received_nft',
                    'avg_value_in_Ether_ever_received_nft', 'minimum_value_in_Ether_ever_sent_nft',
                    'maximum_value_in_Ether_ever_sent_nft', 'avg_value_in_Ether_ever_sent_nft',
                    'total_value_in_Ether_ever_received_nft', 'total_value_in_Ether_ever_sent_nft',
                    'the_number_of_transactions_per_day_nft', 'the_number_of_incoming_transactions_per_day_nft',
                    'the_number_of_outcoming_transactions_per_day_nft', 'the_number_of_incoming_transactions_per_hour_nft',
                    'the_number_of_outcoming_transactions_per_hour_nft', 'the_number_of_incoming_amounts_per_day_nft',
                    'the_number_of_outcoming_amounts_per_day_nft', 'the_total_number_of_amounts_outcoming_plus_incoming_nft',
                    'the_number_of_incoming_amounts_per_hour_nft', 'the_number_of_outcoming_amounts_per_hour_nft',
                    'the_number_of_transactions_per_hour_nft', 'the_number_of_amounts_per_day_nft',
                    'the_number_of_amounts_per_hour_nft',
                    'average_time_between_incoming_transactions_erc1155','average_time_between_outcoming_transactions_erc1155',
                    'time_since_the_first_until_the_last_transaction_erc1155', 'longest_interval_between_two_transactions_erc1155',
                    'shortest_interval_between_two_transactions_erc1155', 'total_number_of_transactions_erc1155',
                    'the_number_of_unique_outcoming_addresses_erc1155', 'the_number_of_unique_incoming_addresses_erc1155',
                    'the_total_number_of_incoming_transactions_erc1155', 'the_total_number_of_outcoming_transactions_erc1155',
                    'the_proportion_of_unique_outcoming_address_transactions_erc1155', 'the_proportion_of_unique_incoming_address_transactions_erc1155',
                    'proportion_of_outcoming_address_transactions_erc1155', 'proportion_of_incoming_address_transactions_erc1155',
                    'minimum_value_in_Ether_ever_received_erc1155', 'maximum_value_in_Ether_ever_received_erc1155',
                    'avg_value_in_Ether_ever_received_erc1155', 'minimum_value_in_Ether_ever_sent_erc1155',
                    'maximum_value_in_Ether_ever_sent_erc1155', 'avg_value_in_Ether_ever_sent_erc1155',
                    'total_value_in_Ether_ever_received_erc1155', 'total_value_in_Ether_ever_sent_erc1155',
                    'the_number_of_transactions_per_day_erc1155', 'the_number_of_incoming_transactions_per_day_erc1155',
                    'the_number_of_outcoming_transactions_per_day_erc1155', 'the_number_of_incoming_transactions_per_hour_erc1155',
                    'the_number_of_outcoming_transactions_per_hour_erc1155', 'the_number_of_incoming_amounts_per_day_erc1155',
                    'the_number_of_outcoming_amounts_per_day_erc1155', 'the_total_number_of_amounts_outcoming_plus_incoming_erc1155',
                    'the_number_of_incoming_amounts_per_hour_erc1155', 'the_number_of_outcoming_amounts_per_hour_erc1155',
                    'the_number_of_transactions_per_hour_erc1155', 'the_number_of_amounts_per_day_erc1155',
                    'the_number_of_amounts_per_hour_erc1155',
                    'average_time_between_incoming_transactions_all', 'average_time_between_outcoming_transactions_all',
                    'time_since_the_first_until_the_last_transaction_all', 'longest_interval_between_two_transactions_all',
                    'shortest_interval_between_two_transactions_all', 'total_number_of_transactions_all',
                    'the_number_of_unique_outcoming_addresses_all', 'the_number_of_unique_incoming_addresses_all',
                    'the_total_number_of_incoming_transactions_all', 'the_total_number_of_outcoming_transactions_all',
                    'the_proportion_of_unique_outcoming_address_transactions_all','the_proportion_of_unique_incoming_address_transactions_all',
                    'proportion_of_outcoming_address_transactions_all', 'proportion_of_incoming_address_transactions_all',
                    'minimum_value_in_Ether_ever_received_all', 'maximum_value_in_Ether_ever_received_all',
                    'avg_value_in_Ether_ever_received_all', 'minimum_value_in_Ether_ever_sent_all',
                    'maximum_value_in_Ether_ever_sent_all', 'avg_value_in_Ether_ever_sent_all',
                    'total_value_in_Ether_ever_received_all', 'total_value_in_Ether_ever_sent_all',
                    'the_number_of_transactions_per_day_all', 'the_number_of_incoming_transactions_per_day_all',
                    'the_number_of_outcoming_transactions_per_day_all', 'the_number_of_incoming_transactions_per_hour_all',
                    'the_number_of_outcoming_transactions_per_hour_all', 'the_number_of_incoming_amounts_per_day_all',
                    'the_number_of_outcoming_amounts_per_day_all', 'the_total_number_of_amounts_outcoming_plus_incoming_all',
                    'the_number_of_incoming_amounts_per_hour_all', 'the_number_of_outcoming_amounts_per_hour_all',
                    'the_number_of_transactions_per_hour_all', 'the_number_of_amounts_per_day_all',
                    'the_number_of_amounts_per_hour_all', 'reverted_numbers_all',
                    'the_propotion_of_normal_transactions_of_all', 'the_propotion_of_normal_incoming_transactions_of_all',
                    'the_propotion_of_normal_incoming_transactions_of_all_transactions', 'the_propotion_of_normal_outcoming_transactions_of_all',
                    'the_propotion_of_normal_outcoming_transactions_of_all_transactions',
                    'the_propotion_of_internal_transactions_of_all', 'the_propotion_of_internal_incoming_transactions_of_all',
                    'the_propotion_of_internal_incoming_transactions_of_all_transactions', 'the_propotion_of_internal_outcoming_transactions_of_all',
                    'the_propotion_of_internal_outcoming_transactions_of_all_transactions',
                    'the_propotion_of_erc20_transactions_of_all','the_propotion_of_erc20_incoming_transactions_of_all',
                    'the_propotion_of_erc20_incoming_transactions_of_all_transactions', 'the_propotion_of_erc20_outcoming_transactions_of_all',
                    'the_propotion_of_erc20_outcoming_transactions_of_all_transactions',
                    'the_propotion_of_nft_transactions_of_all', 'the_propotion_of_nft_incoming_transactions_of_all',
                    'the_propotion_of_nft_incoming_transactions_of_all_transactions', 'the_propotion_of_nft_outcoming_transactions_of_all',
                    'the_propotion_of_nft_outcoming_transactions_of_all_transactions',
                    'the_propotion_of_erc1155_transactions_of_all', 'the_propotion_of_erc1155_incoming_transactions_of_all',
                    'the_propotion_of_erc1155_incoming_transactions_of_all_transactions', 'the_propotion_of_erc1155_outcoming_transactions_of_all',
                    'the_propotion_of_erc1155_outcoming_transactions_of_all_transactions',
                    'the_propotion_of_normal_ether_transfered_of_all', 'the_propotion_of_normal_ether_sent_of_all_sent',
                    'the_propotion_of_normal_ether_sent_of_all_touched', 'the_propotion_of_normal_ether_received_of_all_sent',
                    'the_propotion_of_normal_ether_received_of_all_touched',
                    'the_propotion_of_internal_ether_transfered_of_all','the_propotion_of_internal_ether_sent_of_all_sent',
                    'the_propotion_of_internal_ether_sent_of_all_touched','the_propotion_of_internal_ether_received_of_all_sent',
                    'the_propotion_of_internal_ether_received_of_all_touched',
                    'the_propotion_of_erc20_ether_transfered_of_all', 'the_propotion_of_erc20_ether_sent_of_all_sent',
                    'the_propotion_of_erc20_ether_sent_of_all_touched', 'the_propotion_of_erc20_ether_received_of_all_sent',
                    'the_propotion_of_erc20_ether_received_of_all_touched',
                    'the_propotion_of_nft_ether_transfered_of_all', 'the_propotion_of_nft_ether_sent_of_all_sent',
                    'the_propotion_of_nft_ether_sent_of_all_touched', 'the_propotion_of_nft_ether_received_of_all_sent',
                    'the_propotion_of_nft_ether_received_of_all_touched',
                    'the_propotion_of_erc1155_ether_transfered_of_all', 'the_propotion_of_erc1155_ether_sent_of_all_sent',
                    'the_propotion_of_erc1155_ether_sent_of_all_touched', 'the_propotion_of_erc1155_ether_received_of_all_sent',
                    'the_propotion_of_erc1155_ether_received_of_all_touched']
#feature_set_opcode = ['PUSH1','MSTORE','CALLDATASIZE','LT','PUSH2','JUMPI','CALLDATALOAD','PUSH29','SWAP1','DIV','PUSH4','AND','DUP1','EQ','JUMPDEST','SLOAD','TIMESTAMP','ISZERO','POP','GT','REVERT','CALLVALUE','MUL','JUMP','CALLER','PUSH20','DUP2','ADD','SHA3','DUP3','SSTORE','EXP','PUSH32','DUP4','MLOAD','SWAP2','SUB','LOG3','DUP6','DUP9','CALL','SWAP4','RETURNDATASIZE','RETURNDATACOPY','STOP','DUP5','NOT','SWAP3','RETURN','CALLDATACOPY','DUP7','OR','INVALID','ADDRESS','SWAP6','DUP8','EXTCODESIZE','GAS','LOG1','PUSH6','XOR','CHAINID','SWAP12','SAR','BALANCE','SWAP13','SELFDESTRUCT','SIGNEXTEND','MSIZE','DUP11','PUSH15','SWAP5','SELFBALANCE','DUP12','PUSH7','SWAP7','DELEGATECALL','PUSH10','CREATE','DIFFICULTY','LOG2','SHR','SWAP14','SLT','PUSH18','PUSH12','PUSH22','PUSH5','PUSH13','LOG0','PUSH16','EXTCODECOPY','BYTE','PUSH3','PUSH28','MULMOD','STATICCALL','DUP10','ADDMOD','PUSH9','PUSH25','SWAP9','SWAP8','SWAP10','DUP16','SWAP11','DUP15','PUSH21','DUP13','NUMBER','BLOCKHASH','MOD','CODECOPY','PUSH19','SGT','CODESIZE','CALLCODE','SDIV','GASPRICE','EXTCODEHASH','DUP14','PUSH30','SMOD','PUSH8','MSTORE8','ORIGIN','PUSH17','CREATE2','SHL','COINBASE','GETPC','SWAP15','PUSH27','GASLIMIT','SWAP16','PUSH14','LOG4','PUSH23','PUSH11','PUSH24','PUSH26','PUSH31']
feature_set_opcode_tf_idf = ['push1(opcode)','dup1(opcode)','swap1(opcode)','pop(opcode)','push2(opcode)','dup2(opcode)','add(opcode)','jumpdest(opcode)','mstore(opcode)','and(opcode)','swap2(opcode)','push20(opcode)','iszero(opcode)','mload(opcode)','jumpi(opcode)','dup3(opcode)','jump(opcode)','sub(opcode)','dup4(opcode)','sload(opcode)','revert(opcode)','swap3(opcode)','sha3(opcode)','push4(opcode)','invalid(opcode)','eq(opcode)','exp(opcode)','calldataload(opcode)','dup5(opcode)','shl(opcode)','div(opcode)','mul(opcode)','callvalue(opcode)','returndatasize(opcode)','lt(opcode)','dup6(opcode)','return(opcode)','caller(opcode)','sstore(opcode)','swap4(opcode)','push32(opcode)','stop(opcode)','gt(opcode)','calldatasize(opcode)','codecopy(opcode)','dup7(opcode)','dup9(opcode)','address(opcode)','swap5(opcode)','dup8(opcode)','not(opcode)','call(opcode)','or(opcode)','push8(opcode)','swap6(opcode)','calldatacopy(opcode)','gas(opcode)','extcodesize(opcode)','push29(opcode)','returndatacopy(opcode)','push6(opcode)','delegatecall(opcode)','log1(opcode)','dup12(opcode)','dup15(opcode)','swap8(opcode)','log3(opcode)']
feature_set_ngram_bytecode = ['zmzmzg(bytecode)','mdawma(bytecode)','njawma(bytecode)','njayma(bytecode)','nti2ma(bytecode)','nty1yg(bytecode)','njawmq(bytecode)','mde2ma(bytecode)','mdaxng(bytecode)','njbhma(bytecode)','mtywyq(bytecode)','nja0ma(bytecode)','nta1ma(bytecode)','ode1mg(bytecode)','mtuyng(bytecode)','mdawoa(bytecode)','mdiwma(bytecode)','mjawmq(bytecode)','mdywma(bytecode)','mdiwyq(bytecode)','mjbhma(bytecode)','nwi2ma(bytecode)','mtu2mq(bytecode)','njving(bytecode)','ntyxma(bytecode)','mjywmg(bytecode)','mgewng(bytecode)','mduwnq(bytecode)','yta2ma(bytecode)','mda4ma(bytecode)','mde5ma(bytecode)','ntc2ma(bytecode)','mwiwmw(bytecode)','ytaxyg(bytecode)','mdqwnq(bytecode)','mtywma(bytecode)','mgewmq(bytecode)','mgzknq(bytecode)','zmq1yg(bytecode)','mdaxoq(bytecode)','nzywma(bytecode)','yjywma(bytecode)','mdfima(bytecode)','mdyxma(bytecode)','mdayma(bytecode)','zmzmmq(bytecode)','nznmzg(bytecode)','m2zmzg(bytecode)','njewmq(bytecode)','ota4mq(bytecode)','njawmg(bytecode)','ntq2ma(bytecode)','mgewmw(bytecode)','njewma(bytecode)','ndywma(bytecode)','mdgwnq(bytecode)','oda1na(bytecode)','ota5mq(bytecode)','ndyxma(bytecode)','mtaxnq(bytecode)','ote5ma(bytecode)','mjywma(bytecode)','mdaynq(bytecode)','mtawma(bytecode)','mtq2mq(bytecode)','oda1mq(bytecode)','mdawmg(bytecode)','mde1ng(bytecode)','ntc4ma(bytecode)','nte2ma(bytecode)','ntdmzq(bytecode)','mdywna(bytecode)','zmu1yg(bytecode)','nda1mq(bytecode)','n2zlnq(bytecode)','mtuxnq(bytecode)','mtawnq(bytecode)','mtywmg(bytecode)','mty2ma(bytecode)','mduwng(bytecode)','oda4mw(bytecode)','nwi2mq(bytecode)','mde4na(bytecode)','nwi1ma(bytecode)','odewmq(bytecode)','yjuwnq(bytecode)','yjyxma(bytecode)','mtaxoa(bytecode)','mdqwoa(bytecode)','njayna(bytecode)','ztving(bytecode)','nta1ng(bytecode)','njawna(bytecode)','mduxng(bytecode)','njywma(bytecode)','mduxoa(bytecode)','nta2ma(bytecode)','nte1ng(bytecode)','nte4ma(bytecode)','nte5ma(bytecode)','nda4ma(bytecode)','mdgwzg(bytecode)','odbmza(bytecode)','njewmg(bytecode)','njewna(bytecode)','ota2ma(bytecode)','odiwmq(bytecode)','zdving(bytecode)']
feature_set_ngram_all = ['rfvqmibnu1rpukugufvtsdegmhgyma(op)','tvnut1jfifbvu0gxidb4mjagqure(op)','ufvtsdegmhg0mcbnte9brcbevvax(op)','ufvtsdegmhgymcbbreqgu1dbude(op)','ufvtsdiwidb4zmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzibbtkqgufvtsdiw(op)','qu5eifbvu0gymcawegzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmygqu5e(op)','mhhmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmieforcbqvvnimjagmhhmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzm(op)','ufvtsdegmhgwiervudegukvwrvju(op)','mhgwiervudegukvwrvjuiepvtvbervnu(op)','slvnuekgufvtsdegmhgwiervude(op)','ufvtsdiwidb4zmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzibbtkqgrfvqmg(op)','mhhmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmieforcbevvayie1tve9srq(op)','qu5eiervudigtvnut1jfifbvu0gx(op)','ufvtsdegmhgxifbvu0gxidb4yta(op)','ufvtsdegmhgymcbbreqgufvtsde(op)','ufvtsdegmhgymcbbreqgu1dbudi(op)','u1dbudegrfvqmibnu1rpukugufvtsde(op)','qureifbvu0gxidb4mcbtseez(op)','mhgymcbbreqgu1dbudegrfvqmg(op)','mhgymcbbreqgufvtsdegmhgw(op)','qureifnxqvaxiervudigtvnut1jf(op)','mhgymcbbreqgu1dbudigue9q(op)','u1dbudegufvtsdegmhgymcbbreq(op)','tuxpquqgrfvqmsbtv0fqmibtvui(op)','rfvqmsbtv0fqmibtvuigu1dbude(op)','mhg0mcbnte9brcbevvaxifnxqvay(op)','ue9qifbvu0gxidb4ndagtuxpquq(op)','u1dbudigue9qifbpucbqvvnimq(op)','ue9qifbpucbqvvnimsawedqw(op)','qureifnxqvayifbpucbqt1a(op)','mhg0mcbnte9brcbevvaxiervudm(op)','ufvtsdegmhhhmcbqvvnimsawedi(op)','mhhhmcbqvvnimsawedigrvhq(op)','mhgxifbvu0gxidb4ytagufvtsde(op)','ufvtsdegmhgyievyucbtvui(op)','slvnuerfu1qgq0fmtfzbtfvfiervudegsvnarvjp(op)','ukvwrvjuiepvtvbervnuifbpucbqvvnimg(op)','u1dbudegu1dbudmgu1dbudigu1dbude(op)','slvnuerfu1qgufvtsdegmhg0mcbnte9bra(op)','slvnucbkvu1qrevtvcbqvvnimsawedqw(op)','mhgymcbbreqgu1dbudegu1dbudm(op)','qureifnxqvaxifnxqvazifnxqvay(op)','mhgxifbvu0gxidb4msbqvvnimq(op)','ufvtsdegmhgxifbvu0gxidb4mq(op)','mhgxifbvu0gxidb4ytagu0hm(op)','ue9qifbpucbqt1ague9q(op)','ufvtsdegmhhhmcbtsewgu1vc(op)','u1dbudegukvuvvjoiepvtvbervnuienbtexwquxvrq(op)','rfvqmsbsrvzfulqgslvnuerfu1qgue9q(op)','u1vcifnxqvaxifjfvfvstibkvu1qrevtva(op)','tfqgsvnarvjpifbvu0gyidb4mg(op)','svnarvjpifbvu0gyidb4mibkvu1qsq(op)','u0xpquqgufvtsdegmhgxifbvu0gx(op)','rfvqmibmvcbju1pfuk8gufvtsdi(op)','rfvqmibnte9brcbtv0fqmibnu1rpuku(op)','rfvqncbdt0rfq09qwsbevvayie1mt0fe(op)','ufvtsdegmhgwiervudegtuxpquq(op)','mhgwiervudegtuxpquqgufvtsde(op)','tuxpquqgufvtsdegmhgymcbqvvnimg(op)','q09erunpufkgrfvqmibnte9brcbtv0fqmg(op)','rfvqmsbnte9brcbqvvnimsawediw(op)','u1dbudegrfvqmibmvcbju1pfuk8(op)','ufvtsdegmhg0mcbevvaxie1mt0fe(op)','tvvmifbvu0gxidb4mcbevvax(op)','tuxpquqgu1dbudigtvnut1jfiefera(op)','rfvqmibkvu1qiepvtvbervnuifbvu0gy(op)','mhgyievyucbtvuigqu5e(op)','slvnuerfu1qgq0fmtfzbtfvfifbvu0gyidb4ma(op)','ufvtsdigmhgwiepvtvbjifbvu0gy(op)','q0fmtfzbtfvfifbvu0gyidb4mcbkvu1qsq(op)','slvnuerfu1qgufvtsdegmhg0mcbevvax(op)','rfvqmsbtte9brcbqvvnimsawede(op)','ufvtsdegmhgwifbvu0gxidb4ma(op)','rfvqmybevvayiervudigrfvqmg(op)','rfvqnibevva5iervudqgq0fmta(op)','rfvqmibevva2iervudkgrfvqna(op)','rfvqmibtv0fqmsbtvuigufvtsde(op)','qureifnxqvaxifjfvfvstibkvu1qrevtva(op)','mhgyievyucbtvuigtk9u(op)','rfvqmibevvayiervudygrfvqoq(op)','rfvqmibevvayiervudigrfvqng(op)','u1dbudegrfvqmibtv0fqmsbtvui(op)','tuxpquqgu1dbudegrfvqmibtv0fqmq(op)','svnarvjpifbvu0gyidb4mcbkvu1qsq(op)','ufvtsdigmhgyyzygr0ftifnvqg(op)','ue9qifbpucbqt1agufvtsde(op)','ue9qifbpucbqt1agslvnua(op)','ue9qifbpucbkvu1qiepvtvbervnu(op)','rfvqmsbevvaziervudugq0fmterbvefdt1bz(op)','slvnucbkvu1qrevtvcbtve9qiepvtvbervnu(op)','slvnuekgsu5wquxjrcbkvu1qrevtvcbqt1a(op)','slvnuekgsu5wquxjrcbkvu1qrevtvcbqvvnimg(op)','slvnuerfu1qgue9qifbpucbqt1a(op)','slvnuerfu1qgq0fmtfzbtfvfieltwkvstybqvvnimg(op)','mhg0mcbevvaxie1mt0feifbvu0gx(op)','rfvqmibbreqgrfvqnsbtv0fqmq(op)','mhg0mcbnte9brcbevvaxiervudq(op)','su5wquxjrcbkvu1qrevtvcbqt1ague9q(op)','u1dbudegrelwiervudugtvvm(op)','u1dbudegrfvqmibbreqgrfvqnq(op)','u1dbudcgue9qifbvu0gyidb4mwvm(op)','u1dbudegrfvqnsbbreqgrfvqna(op)','u1dbudegtvnut1jfiervudigtuxpquq(op)','u0xpquqgufvtsdegmhg0mcbevvax(op)','u0hbmybjtlzbteleiervudeyifjfvfvstkrbvefdt1bz(op)','u1dbudegu1dbudmgqureifnxqvay(op)','tvvmiervudygqureiervudu(op)','u1dbudmgu1dbudqgu1dbudigu1dbude(op)','u1dbudmgue9qifbpucbqt1a(op)','u1dbudmgu1dbudigue9qifbpua(op)','u1dbudegu1dbudqgqu5eifnxqvaz(op)','u1dbudigue9qifbpucbtv0fqmw(op)','u1dbudegu1dbudygtvnut1jfiervudu(op)','u1dbudegu1vciefercbevvay(op)','u1dbudegu1vcifbvu0gxidb4mja(op)','u1dbudegue9qiepvtvbervnuifnxqvaz(op)','u1dbudmgu1dbudegu1dbudqgqu5e(op)','u1dbudmgtuxpquqgu1dbudmgu1dbude(op)','u1dbudigrfvqmibtv0fqmsbevva1(op)','u1dbudigsvnarvjpieltwkvstybevvaz(op)','u1dbudigu1dbudegu1dbudmgqure(op)','u1dbudmgrfvqmybtv0fqmsbtvui(op)','u1dbudmgqureifnxqvayiervudi(op)','u1dbudigue9qifbpucbkvu1q(op)','u1dbudegue9qifbvu0gymcawegzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmy(op)','rfvqmsbsrvzfulqgslvnuerfu1qgufvtsdi(op)','q0fmtfzbtfvfiervudegsvnarvjpifbvu0gy(op)','slvnucbkvu1qrevtvcbqvvnimsaweda(op)','ufvtsdegmhg0mcbnte9brcbqvvnimq(op)','svnarvjpiervudegsvnarvjpifbvu0gy(op)','slvnuerfu1qgufvtsdegmhgwiervude(op)','mhgyievyucbtvuigu1dbude(op)','mhgymcbbreqgu1dbudegu1dbudi(op)','qureifnxqvaxifnxqvayifnxqvax(op)','rfvqmsbevva0ifnvqibevvay(op)','tuxpquqgrfvqmsbevva0ifnvqg(op)','ukvwrvjuiepvtvbervnuifbpucbqt1a(op)','ufvtsdigmhgxmdagrvhqifnxqvax(op)','mhgxmdagrvhqifnxqvaxierjvg(op)','u1dbudegufvtsdigmhgxmdagrvhq(op)','u0xpquqgu1dbudegufvtsdigmhgxmda(op)','u1dbudegu0xpquqgu1dbudegufvtsdi(op)','ufvtsdegmhgwifnxqvaxiervudi(op)','mhgwifnxqvaxiervudigtvnut1jf(op)','rfvqmsbsrvzfulqgslvnuerfu1qgufvtsde(op)','ue9qiepvtvagslvnuerfu1qgufvtsde(op)','tuxpquqgufvtsdegmhgymcbqvvnimq(op)','relwifbvu0gymcawegzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmygqu5e(op)','u1dbudegrelwifbvu0gymcawegzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmy(op)','rvhqifnxqvaxierjvibqvvnimja(op)','q0fmtevsifbvu0gymcawegzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmzmygqu5e(op)','ukvuvvjoiepvtvbervnuienbtexwquxvrsbju1pfuk8(op)','u1dbudigu1vcifnxqvaxifjfvfvstg(op)','ufvtsdegmhgwifniqtmgufvtsde(op)','mhgwifniqtmgufvtsdegmhgw(op)']
feature_set_operands_tf_idf = ['0x0(operands)','0x20(operands)','0xffffffffffffffffffffffffffffffffffffffff(operands)','0x40(operands)','0x1(operands)','0xa0(operands)','0x4(operands)','0x2(operands)','0x100(operands)','0x1f(operands)','0x24(operands)','0xffffffff(operands)','0xff(operands)','0x7(operands)','0x461bcd(operands)','0xe5(operands)','0x8c379a000000000000000000000000000000000000000000000000000000000(operands)','0x9(operands)','0x60(operands)','0x6(operands)','0x5(operands)','0xa(operands)','0x44(operands)','0xe0(operands)','0x3(operands)','0x64(operands)','0xf10f6a15e259465232009528ad32ea5743ce152309fc(operands)','0xe659bb22726afd6009b17c3ae23679ed41784f382129f7fb0b2ef6776e0413d9(operands)','0xaf1931c20ee0c11bea17a41bfbbad299b2763bc0(operands)','0x30(operands)','0x87(operands)','0x8fc(operands)','0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef(operands)','0x627a7a723058(operands)','0x25(operands)','0x8(operands)','0x26(operands)','0x100000000000000000000000000000000000000000000000000000000(operands)','0xb(operands)','0x2b(operands)','0x80(operands)','0xa9059cbb(operands)','0x23(operands)','0x2c6(operands)','0x70a08231(operands)','0x8c5be1e5ebec7d5bd14f71427d1e84f3dd0314c0f7b2291e5b200ac8c7c3b925(operands)','0x18160ddd(operands)','0x313ce567(operands)','0x95d89b41(operands)','0x23b872dd(operands)','0x290decd9548b62a8d60345a988386fc84ba6bc95484008f6362f93160ef3e563(operands)','0x107(operands)','0xa8d603(operands)','0xc84ba6bc95484008f6362f93160ef3e5(operands)','0xde0b6b3a7640000(operands)','0x18f(operands)','0xe97dcb62(operands)','0x8da5cb5b(operands)','0xa60f3588(operands)','0x188(operands)','0x6c(operands)','0x13af4035(operands)','0xb69ef8a8(operands)','0x167(operands)','0x9003adfe(operands)','0xd(operands)','0x8ac7230489e80000(operands)','0x129(operands)','0xbf(operands)','0x32(operands)','0xf(operands)','0x16e(operands)','0x50(operands)','0x5a(operands)','0xc0ee0b8a(operands)','0x9a(operands)','0x92(operands)','0x7e(operands)','0x6ea056a9(operands)','0x189(operands)','0x52(operands)','0x1e0(operands)','0x49(operands)','0x104(operands)','0x3c18d31800000000000000000000000000000000000000000000000000000000(operands)','0x3c18d318(operands)','0x17b(operands)','0x1ef(operands)','0x718(operands)','0x8796(operands)','0xa9059cbb00000000000000000000000000000000000000000000000000000000(operands)','0xec(operands)','0x125(operands)','0xfdcf3cf6cbee9677fe38(operands)','0x4b(operands)','0x4d2(operands)','0x1000000000000000000000000(operands)','0x14(operands)','0x25e(operands)','0x260(operands)','0x2ea(operands)','0x464(operands)','0x4de(operands)','0x61(operands)','0x77(operands)','0xffffffffffffffff(operands)','0x6e(operands)','0x3fad9ae0(operands)','0x8d(operands)','0x10b(operands)','0x3853682c(operands)','0x19d(operands)','0xb8(operands)','0x81(operands)','0x32d(operands)','0xfd(operands)','0xc(operands)','0xdd62ed3e(operands)','0xe(operands)','0x6fdde03(operands)','0x95ea7b3(operands)','0x12(operands)','0x10(operands)']

multi_csv = 'multi.csv'
binary_csv = 'binary.csv'
time_list = [' (1hours)', ' (3hours)', ' (6hours)', ' (12hours)', ' (1.0days)', ' (3.0days)', ' (7.0days)', ' (14.0days)', ' (30.0days)', ' (90.0days)', ' (180.0days)', ' (365.0days)', ' (1095.0days)', ' (1825.0days)', ' (3650.0days)']
PGD20_BEST_PARAMS_CSV = "wide_param_search_14600_binary_eps030_pgd20_best_by_model_attack.csv"
PGD10_BEST_PARAMS_CSV = "pgd10_only_param_search_14600_binary_eps030_best_by_model_attack.csv"
BEST_ATTACK_PARAMS = {
    "DT": {
        "FGSM-SNN": {"epsilon": 0.3},
        "PGD-10-SNN": {"epsilon": 0.3, "steps": 10, "alpha_factor": 3.0, "alpha": 0.1},
        "PGD-20-SNN": {"epsilon": 0.3, "steps": 20, "alpha_factor": 9.5, "alpha": 0.031578947368421054},
        "BBA": {"epsilon": 0.3, "max_attempts": 80, "top_k": 160, "perturb_features": 115},
        "DBA-style": {"epsilon": 0.3, "max_features": 105},
    },
    "LightGBM": {
        "FGSM-SNN": {"epsilon": 0.3},
        "PGD-10-SNN": {"epsilon": 0.3, "steps": 10, "alpha_factor": 3.0, "alpha": 0.1},
        "PGD-20-SNN": {"epsilon": 0.3, "steps": 20, "alpha_factor": 17.0, "alpha": 0.01764705882352941},
        "BBA": {"epsilon": 0.3, "max_attempts": 200, "top_k": 130, "perturb_features": 120},
        "DBA-style": {"epsilon": 0.3, "max_features": 35},
    },
    "RF": {
        "FGSM-SNN": {"epsilon": 0.3},
        "PGD-10-SNN": {"epsilon": 0.3, "steps": 10, "alpha_factor": 10.0, "alpha": 0.03},
        "PGD-20-SNN": {"epsilon": 0.3, "steps": 20, "alpha_factor": 19.0, "alpha": 0.015789473684210527},
        "BBA": {"epsilon": 0.3, "max_attempts": 170, "top_k": 280, "perturb_features": 200},
        "DBA-style": {"epsilon": 0.3, "max_features": 175},
    },
    "XGBoost": {
        "FGSM-SNN": {"epsilon": 0.3},
        "PGD-10-SNN": {"epsilon": 0.3, "steps": 10, "alpha_factor": 8.0, "alpha": 0.0375},
        "PGD-20-SNN": {"epsilon": 0.3, "steps": 20, "alpha_factor": 15.5, "alpha": 0.01935483870967742},
        "BBA": {"epsilon": 0.3, "max_attempts": 170, "top_k": 170, "perturb_features": 165},
        "DBA-style": {"epsilon": 0.3, "max_features": 160},
    },
}

code_feature_list = (
    feature_set_opcode_tf_idf
    + feature_set_ngram_bytecode
    + feature_set_ngram_all
    + feature_set_operands_tf_idf
)

def build_feature_cols_for_time(time_suffix):
    transaction_features = []
    for each_feature in without_time_list:
        transaction_features.append(each_feature + time_suffix)
    feature_cols = transaction_features + code_feature_list
    feature_cols = list(dict.fromkeys(feature_cols))
    return feature_cols
dtype_dict = {}

for time_suffix in time_list:
    for each_feature in without_time_list:
        dtype_dict[each_feature + time_suffix] = float
for each_feature in code_feature_list:
    dtype_dict[each_feature] = float
def get_torch_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
DEVICE = get_torch_device()

torch.set_num_threads(TORCH_NUM_THREADS)
torch.set_num_interop_threads(TORCH_INTEROP_THREADS)
print("Using PyTorch device:", DEVICE)

def set_seed(seed=10):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
def clear_runtime_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except RuntimeError:
            pass
        
def clean_feature_columns(df, feature_cols):
    existing_cols = [c for c in feature_cols if c in df.columns]
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        print(f"Warning: {len(missing_cols)} feature columns are missing and will be ignored.")
    if not existing_cols:
        raise ValueError("No valid feature columns found in the dataset.")
    return existing_cols

def prepare_feature_matrix(df, feature_cols):
    feature_cols = clean_feature_columns(df, feature_cols)
    X_df = df[feature_cols].copy()
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    X_df = X_df.clip(lower=-1e30, upper=1e30)
    X_df = X_df.fillna(0)
    return X_df.values.astype(np.float64), feature_cols

def infer_labels(df, category_col="category", task="binary"):
    if category_col not in df.columns:
        raise ValueError(f"Category column '{category_col}' not found in CSV.")
    status_list = df[category_col].unique().tolist()
    label_map = {v: i for i, v in enumerate(status_list)}
    print("Category mapping:", label_map)
    y = df[category_col].apply(lambda x: label_map[x]).values.astype(int)
    benign_label = None
    malicious_labels = []
    for label_name, label_id in label_map.items():
        name_lower = str(label_name).lower().strip()
        if name_lower in ["non-malicious", "non malicious", "benign", "normal"]:
            benign_label = label_id
    if task == "binary":
        malicious_label = None
        for label_name, label_id in label_map.items():
            name_lower = str(label_name).lower().strip()
            if name_lower == "malicious":
                malicious_label = label_id
                break
        if malicious_label is None:
            raise ValueError(f"Cannot identify malicious label from categories: {status_list}")
        malicious_labels = [malicious_label]
    else:
        for label_name, label_id in label_map.items():
            if label_id != benign_label:
                malicious_labels.append(label_id)
    print("Using benign_label =", benign_label)
    print("Using malicious_labels =", malicious_labels)
    return y, benign_label, malicious_labels, label_map

def attack_success_rate(y_true, pred_clean, pred_adv, malicious_labels):
    y_true = np.asarray(y_true).astype(int)
    pred_clean = np.asarray(pred_clean).astype(int)
    pred_adv = np.asarray(pred_adv).astype(int)
    malicious_mask = np.isin(y_true, malicious_labels)
    correct_malicious_mask = malicious_mask & (pred_clean == y_true)
    n_correct = np.sum(correct_malicious_mask)
    if n_correct == 0:
        return 0.0
    n_success = np.sum(correct_malicious_mask & (pred_adv != y_true))
    return n_success / n_correct

def evasion_success_rate(y_true, pred_clean, pred_adv, malicious_labels, benign_label):
    if benign_label is None:
        return np.nan
    y_true = np.asarray(y_true).astype(int)
    pred_clean = np.asarray(pred_clean).astype(int)
    pred_adv = np.asarray(pred_adv).astype(int)
    malicious_mask = np.isin(y_true, malicious_labels)
    correct_malicious_mask = malicious_mask & (pred_clean == y_true)
    n_correct = np.sum(correct_malicious_mask)
    if n_correct == 0:
        return 0.0
    n_evasion = np.sum(correct_malicious_mask & (pred_adv == benign_label))
    return n_evasion / n_correct
def safe_weighted_f1(y_true, pred):
    return f1_score(y_true, pred, average="weighted", zero_division=0)
def _clean_optional_int(value, default):
    if pd.isna(value):
        return default
    return int(value)
def _clean_optional_float(value, default=None):
    if pd.isna(value):
        return default
    return float(value)
def resolve_local_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(SCRIPT_DIR, path)
def load_best_attack_params(
    pgd20_best_csv=PGD20_BEST_PARAMS_CSV,
    pgd10_best_csv=PGD10_BEST_PARAMS_CSV
):
    best_params = {}
    pgd20_best_csv = resolve_local_path(pgd20_best_csv)
    pgd10_best_csv = resolve_local_path(pgd10_best_csv)
    if pgd20_best_csv and os.path.exists(pgd20_best_csv):
        pgd20_df = pd.read_csv(pgd20_best_csv)
        for _, row in pgd20_df.iterrows():
            model_name = row["model"]
            attack_name = row["attack"]
            best_params.setdefault(model_name, {})
            if attack_name == "FGSM-SNN":
                best_params[model_name][attack_name] = {}
            elif attack_name == "PGD-20-SNN":
                best_params[model_name][attack_name] = {
                    "steps": 20,
                    "alpha_factor": _clean_optional_float(row.get("alpha_factor"))
                }
            elif attack_name == "BBA":
                best_params[model_name][attack_name] = {
                    "max_attempts": _clean_optional_int(row.get("max_attempts"), 50),
                    "top_k": _clean_optional_int(row.get("top_k"), 90),
                    "perturb_features": _clean_optional_int(row.get("perturb_features"), 80)
                }
            elif attack_name == "DBA-style":
                best_params[model_name][attack_name] = {
                    "max_features": _clean_optional_int(row.get("max_features"), 5)
                }
    else:
        print("Warning: PGD-20/BBA/DBA best-params CSV not found:", pgd20_best_csv)
    if pgd10_best_csv and os.path.exists(pgd10_best_csv):
        pgd10_df = pd.read_csv(pgd10_best_csv)
        for _, row in pgd10_df.iterrows():
            model_name = row["model"]
            best_params.setdefault(model_name, {})
            best_params[model_name]["PGD-10-SNN"] = {
                "steps": 10,
                "alpha_factor": _clean_optional_float(row.get("alpha_factor"))
            }
    else:
        print("Warning: PGD-10 best-params CSV not found:", pgd10_best_csv)
    print("\nLoaded best attack parameters:")
    for model_name in sorted(best_params):
        print(model_name, best_params[model_name])
    return best_params
class SurrogateNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)
def train_surrogate_snn(
    X_train,
    target_model,
    num_classes,
    epochs=80,
    batch_size=256,
    lr=1e-3
):
    target_labels = target_model.predict(X_train).astype(int)
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(target_labels, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(DEVICE.type == "cuda")
    )
    snn = SurrogateNN(
        input_dim=X_train.shape[1],
        num_classes=num_classes
    ).to(DEVICE)
    optimizer = optim.Adam(snn.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    snn.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            logits = snn(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 20 == 0:
            loader_len = len(loader)
            if loader_len == 0:
                loader_len = 1
            avg_loss = total_loss / loader_len
            print(f"    [SNN] epoch {epoch + 1}/{epochs}, loss={avg_loss:.4f}")
    return snn
def surrogate_agreement(snn, target_model, X):
    snn.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        logits = snn(X_tensor)
        pred_snn = torch.argmax(logits, dim=1).cpu().numpy()
    pred_target = target_model.predict(X).astype(int)
    return accuracy_score(pred_target, pred_snn)

def fgsm_snn_attack(
    snn,
    X,
    y,
    epsilon,
    feature_mask=None
):
    snn.eval()
    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
        device=DEVICE
    ).detach().requires_grad_(True)
    y_tensor = torch.tensor(
        y.astype(int),
        dtype=torch.long,
        device=DEVICE
    )
    criterion = nn.CrossEntropyLoss()
    logits = snn(X_tensor)
    loss = criterion(logits, y_tensor)
    snn.zero_grad()
    loss.backward()
    grad_sign = X_tensor.grad.data.sign()
    if feature_mask is not None:
        mask = torch.tensor(feature_mask, dtype=torch.float32, device=DEVICE)
        grad_sign = grad_sign * mask
    X_adv = X_tensor + epsilon * grad_sign
    snn.zero_grad(set_to_none=True)
    return X_adv.detach().cpu().numpy()

def pgd_snn_attack(
    snn,
    X,
    y,
    epsilon,
    alpha,
    steps,
    feature_mask=None
):
    snn.eval()
    X_orig = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    X_adv = X_orig.clone().detach()
    X_adv = X_adv + torch.empty_like(X_adv).uniform_(-epsilon, epsilon)
    y_tensor = torch.tensor(
        y.astype(int),
        dtype=torch.long,
        device=DEVICE
    )
    criterion = nn.CrossEntropyLoss()
    if feature_mask is not None:
        mask = torch.tensor(feature_mask, dtype=torch.float32, device=DEVICE)
    else:
        mask = None
    for _ in range(steps):
        X_adv = X_adv.detach().requires_grad_(True)
        logits = snn(X_adv)
        loss = criterion(logits, y_tensor)
        snn.zero_grad()
        loss.backward()
        grad_sign = X_adv.grad.data.sign()
        if mask is not None:
            grad_sign = grad_sign * mask
        X_adv = X_adv.detach() + alpha * grad_sign
        delta = torch.clamp(
            X_adv - X_orig,
            min=-epsilon,
            max=epsilon
        )
        X_adv = (X_orig + delta).detach()
    snn.zero_grad(set_to_none=True)
    return X_adv.detach().cpu().numpy()

def fgsm_snn_targeted_attack(
    snn,
    X,
    target_label,
    epsilon,
    attack_indices,
    feature_mask=None
):
    X_adv = X.copy()
    if len(attack_indices) == 0:
        return X_adv
    snn.eval()
    X_attack = torch.tensor(
        X[attack_indices],
        dtype=torch.float32,
        device=DEVICE
    ).detach().requires_grad_(True)
    y_target = torch.full(
        (len(attack_indices),),
        int(target_label),
        dtype=torch.long,
        device=DEVICE
    )
    criterion = nn.CrossEntropyLoss()
    logits = snn(X_attack)
    loss = criterion(logits, y_target)
    snn.zero_grad()
    loss.backward()
    grad_sign = X_attack.grad.data.sign()
    if feature_mask is not None:
        mask = torch.tensor(feature_mask, dtype=torch.float32, device=DEVICE)
        grad_sign = grad_sign * mask
    X_targeted = X_attack - epsilon * grad_sign
    X_adv[attack_indices] = X_targeted.detach().cpu().numpy()
    snn.zero_grad(set_to_none=True)
    return X_adv

def pgd_snn_targeted_attack(
    snn,
    X,
    target_label,
    epsilon,
    alpha,
    steps,
    attack_indices,
    feature_mask=None
):
    X_adv_full = X.copy()
    if len(attack_indices) == 0:
        return X_adv_full
    snn.eval()
    X_orig = torch.tensor(
        X[attack_indices],
        dtype=torch.float32,
        device=DEVICE
    )
    X_adv = X_orig.clone().detach()
    X_adv = X_adv + torch.empty_like(X_adv).uniform_(-epsilon, epsilon)
    y_target = torch.full(
        (len(attack_indices),),
        int(target_label),
        dtype=torch.long,
        device=DEVICE
    )
    criterion = nn.CrossEntropyLoss()
    if feature_mask is not None:
        mask = torch.tensor(feature_mask, dtype=torch.float32, device=DEVICE)
    else:
        mask = None
    for _ in range(steps):
        X_adv = X_adv.detach().requires_grad_(True)
        logits = snn(X_adv)
        loss = criterion(logits, y_target)
        snn.zero_grad()
        loss.backward()
        grad_sign = X_adv.grad.data.sign()
        if mask is not None:
            grad_sign = grad_sign * mask
        X_adv = X_adv.detach() - alpha * grad_sign
        delta = torch.clamp(
            X_adv - X_orig,
            min=-epsilon,
            max=epsilon
        )
        X_adv = (X_orig + delta).detach()
    X_adv_full[attack_indices] = X_adv.detach().cpu().numpy()
    snn.zero_grad(set_to_none=True)
    return X_adv_full

def _bba_one_sample(
    i,
    model,
    x0,
    y_i,
    pred_clean_i,
    top_features,
    epsilon,
    max_attempts,
    perturb_features,
    random_state
):
    if pred_clean_i != y_i:
        return x0.copy()
    rng = np.random.default_rng(random_state)
    candidates = np.repeat(
        x0.reshape(1, -1),
        repeats=max_attempts,
        axis=0
    )
    for attempt_id in range(max_attempts):
        selected = rng.choice(
            top_features,
            size=perturb_features,
            replace=False
        )
        directions = rng.choice([-1.0, 1.0], size=perturb_features)
        magnitudes = rng.uniform(0.0, epsilon, size=perturb_features)
        candidates[attempt_id, selected] = (
            candidates[attempt_id, selected] + directions * magnitudes
        )
    preds = model.predict(candidates).astype(int)
    success_indices = np.where(preds != y_i)[0]
    if len(success_indices) > 0:
        return candidates[success_indices[0]].copy()
    return x0.copy()
def _bba_chunk(
    model,
    X,
    y,
    pred_clean,
    indices,
    top_features,
    epsilon,
    max_attempts,
    perturb_features,
    random_state
):
    chunk_adv = []
    for i in indices:
        chunk_adv.append(
            _bba_one_sample(
                i,
                model,
                X[i],
                y[i],
                pred_clean[i],
                top_features,
                epsilon,
                max_attempts,
                perturb_features,
                random_state
            )
        )
    return indices, np.asarray(chunk_adv, dtype=X.dtype)

def bba_attack(
    model,
    X,
    y,
    epsilon,
    max_attempts=50,
    top_k=90,
    perturb_features=80,
    random_state=10,
    n_jobs=N_JOBS
):
    pred_clean = model.predict(X).astype(int)
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
    else:
        importances = np.ones(X.shape[1])
    top_k = min(top_k, X.shape[1])
    top_features = np.argsort(importances)[::-1][:top_k]
    perturb_features = min(perturb_features, top_k)
    attack_indices = np.where(pred_clean == y)[0]
    if len(attack_indices) == 0:
        return X.copy()
    effective_n_jobs = min(max(1, n_jobs), len(attack_indices))
    print(
        f"BBA multiprocessing: samples={len(attack_indices)}, "
        f"n_jobs={effective_n_jobs}, max_attempts={max_attempts}"
    )
    if effective_n_jobs == 1:
        X_adv = X.copy()
        for i in attack_indices:
            X_adv[i] = _bba_one_sample(
                i,
                model,
                X[i],
                y[i],
                pred_clean[i],
                top_features,
                epsilon,
                max_attempts,
                perturb_features,
                random_state
            )
        return X_adv
    else:
        chunks = [
            chunk.astype(int)
            for chunk in np.array_split(attack_indices, effective_n_jobs)
            if len(chunk) > 0
        ]
        chunk_results = Parallel(
            n_jobs=effective_n_jobs,
            backend="loky",
            batch_size=1,
            pre_dispatch=effective_n_jobs
        )(
            delayed(_bba_chunk)(
                model,
                X,
                y,
                pred_clean,
                chunk,
                top_features,
                epsilon,
                max_attempts,
                perturb_features,
                random_state
            )
            for chunk in chunks
        )
        X_adv = X.copy()
        for indices, chunk_adv in chunk_results:
            X_adv[indices] = chunk_adv
        return X_adv
    
def _bba_targeted_one_sample(
    model,
    x0,
    target_label,
    top_features,
    epsilon,
    max_attempts,
    perturb_features,
    random_state
):
    rng = np.random.default_rng(random_state)
    candidates = np.repeat(
        x0.reshape(1, -1),
        repeats=max_attempts,
        axis=0
    )
    for attempt_id in range(max_attempts):
        selected = rng.choice(
            top_features,
            size=perturb_features,
            replace=False
        )
        directions = rng.choice([-1.0, 1.0], size=perturb_features)
        magnitudes = rng.uniform(0.0, epsilon, size=perturb_features)
        candidates[attempt_id, selected] = (
            candidates[attempt_id, selected] + directions * magnitudes
        )
    preds = model.predict(candidates).astype(int)
    success_indices = np.where(preds == target_label)[0]
    if len(success_indices) > 0:
        return candidates[success_indices[0]].copy()
    return x0.copy()

def _bba_targeted_chunk(
    model,
    X,
    indices,
    target_label,
    top_features,
    epsilon,
    max_attempts,
    perturb_features,
    random_state
):
    chunk_adv = []
    for i in indices:
        chunk_adv.append(
            _bba_targeted_one_sample(
                model,
                X[i],
                target_label,
                top_features,
                epsilon,
                max_attempts,
                perturb_features,
                random_state
            )
        )
    return indices, np.asarray(chunk_adv, dtype=X.dtype)

def bba_targeted_attack(
    model,
    X,
    target_label,
    epsilon,
    attack_indices,
    max_attempts=50,
    top_k=90,
    perturb_features=80,
    random_state=10,
    n_jobs=N_JOBS
):
    X_adv = X.copy()
    if len(attack_indices) == 0:
        return X_adv
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
    else:
        importances = np.ones(X.shape[1])
    top_k = min(top_k, X.shape[1])
    top_features = np.argsort(importances)[::-1][:top_k]
    perturb_features = min(perturb_features, top_k)
    effective_n_jobs = min(max(1, n_jobs), len(attack_indices))
    print(
        f"Targeted BBA multiprocessing: samples={len(attack_indices)}, "
        f"n_jobs={effective_n_jobs}, max_attempts={max_attempts}"
    )
    if effective_n_jobs == 1:
        for i in attack_indices:
            X_adv[i] = _bba_targeted_one_sample(
                model,
                X[i],
                target_label,
                top_features,
                epsilon,
                max_attempts,
                perturb_features,
                random_state
            )
        return X_adv
    chunks = [
        chunk.astype(int)
        for chunk in np.array_split(attack_indices, effective_n_jobs)
        if len(chunk) > 0
    ]
    chunk_results = Parallel(
        n_jobs=effective_n_jobs,
        backend="loky",
        batch_size=1,
        pre_dispatch=effective_n_jobs
    )(
        delayed(_bba_targeted_chunk)(
            model,
            X,
            chunk,
            target_label,
            top_features,
            epsilon,
            max_attempts,
            perturb_features,
            random_state
        )
        for chunk in chunks
    )
    for indices, chunk_adv in chunk_results:
        X_adv[indices] = chunk_adv
    return X_adv

def _dba_one_sample(
    model,
    x0,
    y_i,
    pred_clean_i,
    top_features,
    epsilon,
):
    if pred_clean_i != y_i:
        return x0.copy()
    best = x0.copy()
    for j in top_features:
        candidates = np.repeat(
            best.reshape(1, -1),
            repeats=4,
            axis=0
        )
        candidates[0, j] = best[j] + epsilon
        candidates[1, j] = best[j] - epsilon
        candidates[2, j] = best[j] + 2 * epsilon
        candidates[3, j] = best[j] - 2 * epsilon
        preds = model.predict(candidates).astype(int)
        success_indices = np.where(preds != y_i)[0]
        if len(success_indices) > 0:
            return candidates[success_indices[0]].copy()
    return best
def _dba_chunk(
    model,
    X,
    y,
    pred_clean,
    indices,
    top_features,
    epsilon
):
    chunk_adv = []
    for i in indices:
        chunk_adv.append(
            _dba_one_sample(
                model,
                X[i],
                y[i],
                pred_clean[i],
                top_features,
                epsilon
            )
        )
    return indices, np.asarray(chunk_adv, dtype=X.dtype)
def dba_style_attack(
    model,
    X,
    y,
    epsilon,
    max_features=5,
    n_jobs=N_JOBS
):
    pred_clean = model.predict(X).astype(int)
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
    else:
        importances = np.ones(X.shape[1])
    max_features = min(max_features, X.shape[1])
    top_features = np.argsort(importances)[::-1][:max_features]
    attack_indices = np.where(pred_clean == y)[0]
    if len(attack_indices) == 0:
        return X.copy()
    effective_n_jobs = min(max(1, n_jobs), len(attack_indices))
    print(
        f"DBA multiprocessing: samples={len(attack_indices)}, "
        f"n_jobs={effective_n_jobs}, max_features={max_features}"
    )
    if effective_n_jobs == 1:
        X_adv = X.copy()
        for i in attack_indices:
            X_adv[i] = _dba_one_sample(
                model,
                X[i],
                y[i],
                pred_clean[i],
                top_features,
                epsilon
            )
        return X_adv
    chunks = [
        chunk.astype(int)
        for chunk in np.array_split(attack_indices, effective_n_jobs)
        if len(chunk) > 0
    ]
    chunk_results = Parallel(
        n_jobs=effective_n_jobs,
        backend="loky",
        batch_size=1,
        pre_dispatch=effective_n_jobs
    )(
        delayed(_dba_chunk)(
            model,
            X,
            y,
            pred_clean,
            chunk,
            top_features,
            epsilon
        )
        for chunk in chunks
    )
    X_adv = X.copy()
    for indices, chunk_adv in chunk_results:
        X_adv[indices] = chunk_adv
    return X_adv

def _dba_targeted_one_sample(
    model,
    x0,
    target_label,
    top_features,
    epsilon
):
    best = x0.copy()
    for j in top_features:
        candidates = np.repeat(
            best.reshape(1, -1),
            repeats=4,
            axis=0
        )
        candidates[0, j] = best[j] + epsilon
        candidates[1, j] = best[j] - epsilon
        candidates[2, j] = best[j] + 2 * epsilon
        candidates[3, j] = best[j] - 2 * epsilon
        preds = model.predict(candidates).astype(int)
        success_indices = np.where(preds == target_label)[0]
        if len(success_indices) > 0:
            return candidates[success_indices[0]].copy()
    return best

def _dba_targeted_chunk(
    model,
    X,
    indices,
    target_label,
    top_features,
    epsilon
):
    chunk_adv = []
    for i in indices:
        chunk_adv.append(
            _dba_targeted_one_sample(
                model,
                X[i],
                target_label,
                top_features,
                epsilon
            )
        )
    return indices, np.asarray(chunk_adv, dtype=X.dtype)

def dba_style_targeted_attack(
    model,
    X,
    target_label,
    epsilon,
    attack_indices,
    max_features=5,
    n_jobs=N_JOBS
):
    X_adv = X.copy()
    if len(attack_indices) == 0:
        return X_adv
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
    else:
        importances = np.ones(X.shape[1])
    max_features = min(max_features, X.shape[1])
    top_features = np.argsort(importances)[::-1][:max_features]
    effective_n_jobs = min(max(1, n_jobs), len(attack_indices))
    print(
        f"Targeted DBA multiprocessing: samples={len(attack_indices)}, "
        f"n_jobs={effective_n_jobs}, max_features={max_features}"
    )
    if effective_n_jobs == 1:
        for i in attack_indices:
            X_adv[i] = _dba_targeted_one_sample(
                model,
                X[i],
                target_label,
                top_features,
                epsilon
            )
        return X_adv
    chunks = [
        chunk.astype(int)
        for chunk in np.array_split(attack_indices, effective_n_jobs)
        if len(chunk) > 0
    ]
    chunk_results = Parallel(
        n_jobs=effective_n_jobs,
        backend="loky",
        batch_size=1,
        pre_dispatch=effective_n_jobs
    )(
        delayed(_dba_targeted_chunk)(
            model,
            X,
            chunk,
            target_label,
            top_features,
            epsilon
        )
        for chunk in chunks
    )
    for indices, chunk_adv in chunk_results:
        X_adv[indices] = chunk_adv
    return X_adv

def build_models(num_classes, seed=10):
    models = {
        "RF": RandomForestClassifier(
            criterion="gini",
            max_features="sqrt",
            min_samples_leaf=1,
            min_samples_split=2,
            n_estimators=115,
            n_jobs=1,
            random_state=seed
        ),
        "DT": DecisionTreeClassifier(
            criterion="log_loss",
            max_depth=13,
            min_samples_leaf=3,
            min_samples_split=17,
            splitter="best",
            random_state=seed
        ),
        "XGBoost": xgb.XGBClassifier(
            gamma=0,
            max_delta_step=0,
            max_depth=7,
            min_child_weight=5,
            n_jobs=1,
            random_state=seed,
            tree_method="hist"
        ),
        "LightGBM": lgb.LGBMClassifier(
            max_delta_step=8,
            max_depth=8,
            num_leaves=21,
            n_jobs=1,
            random_state=seed,
            verbose=-1
        )
    }
    return models

def run_attack_experiment_for_task(
    csv_path,
    dtype_dict,
    feature_cols,
    task_name,
    category_col="category",
    epsilons=(0.05, 0.10, 0.20, 0.30),
    pgd_steps_list=(10, 20),
    seed=10,
    n_splits=4,
    snn_epochs=80,
    output_prefix="attack_results",
    attack_params=None
):
    set_seed(seed)
    print("\n" + "=" * 80)
    print(f"Running task: {task_name}")
    print("=" * 80)
    print("Loading dataset:", csv_path)
    df = pd.read_csv(csv_path, dtype=dtype_dict)
    df = df.replace([np.inf, -np.inf], np.nan)
    task_type = "binary" if task_name.lower().startswith("binary") else "multi"
    y, benign_label, malicious_labels, label_map = infer_labels(
        df,
        category_col=category_col,
        task=task_type
    )
    X, used_feature_cols = prepare_feature_matrix(df, feature_cols)
    num_classes = len(np.unique(y))
    print("Feature matrix shape:", X.shape)
    print("Number of used features:", len(used_feature_cols))
    print("Label shape:", y.shape)
    print("Num classes:", num_classes)
    print("X finite before split:", np.isfinite(X).all())
    print("X max before scaling:", np.nanmax(X))
    print("X min before scaling:", np.nanmin(X))
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed
    )
    all_results = []
    for fold_id, (train_index, test_index) in enumerate(skf.split(X, y), start=1):
        print(f"\n========== {task_name} | Fold {fold_id} ==========")
        X_train_raw = X[train_index]
        X_test_raw = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)
        X_train_scaled = np.nan_to_num(
            X_train_scaled,
            nan=0.0,
            posinf=1e6,
            neginf=-1e6
        ).astype(np.float32)
        X_test_scaled = np.nan_to_num(
            X_test_scaled,
            nan=0.0,
            posinf=1e6,
            neginf=-1e6
        ).astype(np.float32)
        print("X_train finite after scaling:", np.isfinite(X_train_scaled).all())
        print("X_test finite after scaling:", np.isfinite(X_test_scaled).all())
        models = build_models(num_classes=num_classes, seed=seed)
        for model_name, model in models.items():
            print(f"\n--- Task: {task_name} | Model: {model_name} ---")
            model_attack_params = {}
            if attack_params is not None:
                model_attack_params = attack_params.get(model_name, {})
            if model_attack_params:
                print("Using attack params:", model_attack_params)
            model.fit(X_train_scaled, y_train)
            pred_clean = model.predict(X_test_scaled).astype(int)
            clean_acc = accuracy_score(y_test, pred_clean)
            clean_f1 = safe_weighted_f1(y_test, pred_clean)
            print(f"Clean Accuracy: {clean_acc:.4f}")
            print(f"Clean Weighted F1: {clean_f1:.4f}")
            snn = train_surrogate_snn(
                X_train=X_train_scaled,
                target_model=model,
                num_classes=num_classes,
                epochs=snn_epochs,
                batch_size=256,
                lr=1e-3
            )
            agreement = surrogate_agreement(
                snn=snn,
                target_model=model,
                X=X_test_scaled
            )
            print(f"SNN Agreement: {agreement:.4f}")
            for eps in epsilons:
                print(
                    f"\nEpsilon = {eps:.2f} "
                    "(CSV parameters fixed except epsilon)"
                )
                rows = []
                fgsm_params = model_attack_params.get("FGSM-SNN", {})
                fgsm_epsilon = eps
                print(f"Running FGSM-SNN, epsilon={fgsm_epsilon:.2f}")
                X_fgsm = fgsm_snn_attack(
                    snn=snn,
                    X=X_test_scaled,
                    y=y_test,
                    epsilon=fgsm_epsilon
                )
                pred_fgsm = model.predict(X_fgsm).astype(int)
                fgsm_acc = accuracy_score(y_test, pred_fgsm)
                fgsm_f1 = safe_weighted_f1(y_test, pred_fgsm)
                fgsm_asr = attack_success_rate(
                    y_true=y_test,
                    pred_clean=pred_clean,
                    pred_adv=pred_fgsm,
                    malicious_labels=malicious_labels
                )
                fgsm_evasion_asr = evasion_success_rate(
                    y_true=y_test,
                    pred_clean=pred_clean,
                    pred_adv=pred_fgsm,
                    malicious_labels=malicious_labels,
                    benign_label=benign_label
                )
                rows.append({
                    "task": task_name,
                    "fold": fold_id,
                    "model": model_name,
                    "attack": "FGSM-SNN",
                    "epsilon": fgsm_epsilon,
                    "clean_acc": clean_acc,
                    "clean_f1": clean_f1,
                    "adv_acc": fgsm_acc,
                    "adv_f1": fgsm_f1,
                    "acc_drop": clean_acc - fgsm_acc,
                    "asr": fgsm_asr,
                    "evasion_asr": fgsm_evasion_asr,
                    "snn_agreement": agreement,
                    "num_classes": num_classes,
                    "num_features": len(used_feature_cols)
                })
                print(
                    f"FGSM-SNN | adv_acc={fgsm_acc:.4f}, "
                    f"ASR={fgsm_asr:.4f}, Evasion={fgsm_evasion_asr:.4f}"
                )
                for pgd_steps in pgd_steps_list:
                    pgd_param_key = f"PGD-{pgd_steps}-SNN"
                    pgd_params = model_attack_params.get(pgd_param_key, {})
                    if model_attack_params:
                        pgd_epsilon = eps
                        pgd_steps_for_attack = int(pgd_params["steps"])
                        alpha_factor = pgd_params["alpha_factor"]
                        alpha = pgd_params["alpha"]
                    else:
                        pgd_epsilon = eps
                        pgd_steps_for_attack = pgd_steps
                        alpha_factor = 5.0
                        alpha = pgd_epsilon / alpha_factor
                    print(
                        f"Running PGD-{pgd_steps_for_attack}-SNN, "
                        f"epsilon={pgd_epsilon:.2f}, "
                        f"alpha_factor={alpha_factor}, alpha={alpha:.6f}"
                    )
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                    X_pgd = pgd_snn_attack(
                        snn=snn,
                        X=X_test_scaled,
                        y=y_test,
                        epsilon=pgd_epsilon,
                        alpha=alpha,
                        steps=pgd_steps_for_attack
                    )
                    pred_pgd = model.predict(X_pgd).astype(int)
                    pgd_acc = accuracy_score(y_test, pred_pgd)
                    pgd_f1 = safe_weighted_f1(y_test, pred_pgd)
                    pgd_asr = attack_success_rate(
                        y_true=y_test,
                        pred_clean=pred_clean,
                        pred_adv=pred_pgd,
                        malicious_labels=malicious_labels
                    )
                    pgd_evasion_asr = evasion_success_rate(
                        y_true=y_test,
                        pred_clean=pred_clean,
                        pred_adv=pred_pgd,
                        malicious_labels=malicious_labels,
                        benign_label=benign_label
                    )
                    rows.append({
                        "task": task_name,
                        "fold": fold_id,
                        "model": model_name,
                        "attack": f"PGD-{pgd_steps_for_attack}-SNN",
                        "epsilon": pgd_epsilon,
                        "steps": pgd_steps_for_attack,
                        "alpha_factor": alpha_factor,
                        "alpha": alpha,
                        "clean_acc": clean_acc,
                        "clean_f1": clean_f1,
                        "adv_acc": pgd_acc,
                        "adv_f1": pgd_f1,
                        "acc_drop": clean_acc - pgd_acc,
                        "asr": pgd_asr,
                        "evasion_asr": pgd_evasion_asr,
                        "snn_agreement": agreement,
                        "num_classes": num_classes,
                        "num_features": len(used_feature_cols)
                    })
                    print(
                        f"PGD-{pgd_steps_for_attack}-SNN | adv_acc={pgd_acc:.4f}, "
                        f"ASR={pgd_asr:.4f}, Evasion={pgd_evasion_asr:.4f}"
                    )
                bba_params = model_attack_params.get("BBA", {})
                bba_epsilon = eps
                bba_max_attempts = bba_params.get("max_attempts", 50)
                bba_top_k = bba_params.get("top_k", 90)
                bba_perturb_features = bba_params.get("perturb_features", 80)
                print(
                    f"Running BBA, epsilon={bba_epsilon:.2f}, "
                    f"max_attempts={bba_max_attempts}, top_k={bba_top_k}, "
                    f"perturb_features={bba_perturb_features}"
                )
                X_bba = bba_attack(
                    model=model,
                    X=X_test_scaled,
                    y=y_test,
                    epsilon=bba_epsilon,
                    max_attempts=bba_max_attempts,
                    top_k=bba_top_k,
                    perturb_features=bba_perturb_features,
                    random_state=seed,
                    n_jobs=N_JOBS
                )
                pred_bba = model.predict(X_bba).astype(int)
                bba_acc = accuracy_score(y_test, pred_bba)
                bba_f1 = safe_weighted_f1(y_test, pred_bba)
                bba_asr = attack_success_rate(
                    y_true=y_test,
                    pred_clean=pred_clean,
                    pred_adv=pred_bba,
                    malicious_labels=malicious_labels
                )
                bba_evasion_asr = evasion_success_rate(
                    y_true=y_test,
                    pred_clean=pred_clean,
                    pred_adv=pred_bba,
                    malicious_labels=malicious_labels,
                    benign_label=benign_label
                )
                rows.append({
                    "task": task_name,
                    "fold": fold_id,
                    "model": model_name,
                    "attack": "BBA",
                    "epsilon": bba_epsilon,
                    "max_attempts": bba_max_attempts,
                    "top_k": bba_top_k,
                    "perturb_features": bba_perturb_features,
                    "clean_acc": clean_acc,
                    "clean_f1": clean_f1,
                    "adv_acc": bba_acc,
                    "adv_f1": bba_f1,
                    "acc_drop": clean_acc - bba_acc,
                    "asr": bba_asr,
                    "evasion_asr": bba_evasion_asr,
                    "snn_agreement": agreement,
                    "num_classes": num_classes,
                    "num_features": len(used_feature_cols)
                })
                print(
                    f"BBA | adv_acc={bba_acc:.4f}, "
                    f"ASR={bba_asr:.4f}, Evasion={bba_evasion_asr:.4f}"
                )
                dba_params = model_attack_params.get("DBA-style", {})
                dba_epsilon = eps
                dba_max_features = dba_params.get("max_features", 5)
                print(
                    f"Running DBA-style, epsilon={dba_epsilon:.2f}, "
                    f"max_features={dba_max_features}"
                )
                X_dba = dba_style_attack(
                    model=model,
                    X=X_test_scaled,
                    y=y_test,
                    epsilon=dba_epsilon,
                    max_features=dba_max_features,
                    n_jobs=N_JOBS
                )
                pred_dba = model.predict(X_dba).astype(int)
                dba_acc = accuracy_score(y_test, pred_dba)
                dba_f1 = safe_weighted_f1(y_test, pred_dba)
                dba_asr = attack_success_rate(
                    y_true=y_test,
                    pred_clean=pred_clean,
                    pred_adv=pred_dba,
                    malicious_labels=malicious_labels
                )
                dba_evasion_asr = evasion_success_rate(
                    y_true=y_test,
                    pred_clean=pred_clean,
                    pred_adv=pred_dba,
                    malicious_labels=malicious_labels,
                    benign_label=benign_label
                )
                rows.append({
                    "task": task_name,
                    "fold": fold_id,
                    "model": model_name,
                    "attack": "DBA-style",
                    "epsilon": dba_epsilon,
                    "max_features": dba_max_features,
                    "clean_acc": clean_acc,
                    "clean_f1": clean_f1,
                    "adv_acc": dba_acc,
                    "adv_f1": dba_f1,
                    "acc_drop": clean_acc - dba_acc,
                    "asr": dba_asr,
                    "evasion_asr": dba_evasion_asr,
                    "snn_agreement": agreement,
                    "num_classes": num_classes,
                    "num_features": len(used_feature_cols)
                })
                print(
                    f"DBA-style | adv_acc={dba_acc:.4f}, "
                    f"ASR={dba_asr:.4f}, Evasion={dba_evasion_asr:.4f}"
                )
                all_results.extend(rows)
                del rows
                del X_fgsm, pred_fgsm
                del X_pgd, pred_pgd
                del X_bba, pred_bba
                del X_dba, pred_dba
                clear_runtime_cache()
    results_df = pd.DataFrame(all_results)
    result_file = f"{output_prefix}_{task_name}_fold_results.csv"
    summary_file = f"{output_prefix}_{task_name}_summary.csv"
    results_df.to_csv(result_file, index=False)
    summary = results_df.groupby(
        ["task", "model", "attack", "epsilon"],
        as_index=False
    ).agg({
        "clean_acc": "mean",
        "clean_f1": "mean",
        "adv_acc": "mean",
        "adv_f1": "mean",
        "acc_drop": "mean",
        "asr": "mean",
        "evasion_asr": "mean",
        "snn_agreement": "mean",
        "num_features": "mean"
    })
    summary.to_csv(summary_file, index=False)
    print("\n===== Summary:", task_name, "=====")
    print(summary)
    print("\nSaved:")
    print(result_file)
    print(summary_file)
    return results_df, summary

def build_targeted_non_malicious_row(
    task_name,
    fold_id,
    model_name,
    attack_name,
    epsilon,
    params,
    y_test,
    pred_clean,
    pred_adv,
    clean_acc,
    clean_f1,
    agreement,
    malicious_labels,
    benign_label,
    num_classes,
    num_features,
    num_targetable
):
    adv_acc = accuracy_score(y_test, pred_adv)
    adv_f1 = safe_weighted_f1(y_test, pred_adv)
    targeted_evasion_asr = evasion_success_rate(
        y_true=y_test,
        pred_clean=pred_clean,
        pred_adv=pred_adv,
        malicious_labels=malicious_labels,
        benign_label=benign_label
    )
    row = {
        "task": task_name,
        "fold": fold_id,
        "model": model_name,
        "attack": attack_name,
        "epsilon": epsilon,
        "target_label": benign_label,
        "clean_acc": clean_acc,
        "clean_f1": clean_f1,
        "adv_acc": adv_acc,
        "adv_f1": adv_f1,
        "acc_drop": clean_acc - adv_acc,
        "targeted_evasion_asr": targeted_evasion_asr,
        "snn_agreement": agreement,
        "num_classes": num_classes,
        "num_features": num_features,
        "num_targetable": num_targetable
    }
    row.update(params)
    print(
        f"{attack_name} | adv_acc={adv_acc:.4f}, "
        f"TargetedEvasion={targeted_evasion_asr:.4f}"
    )
    return row

def run_multi_targeted_non_malicious_attack_experiment(
    csv_path,
    dtype_dict,
    feature_cols,
    category_col="category",
    epsilons=(0.05, 0.10, 0.20, 0.30),
    pgd_steps_list=(10, 20),
    seed=10,
    n_splits=4,
    snn_epochs=80,
    output_prefix="multi_targeted_non_malicious_attack",
    attack_params=None
):
    set_seed(seed)
    task_name = "multi_targeted_non_malicious"
    print("\n" + "=" * 80)
    print("Running task:", task_name)
    print("=" * 80)
    print("Loading dataset:", csv_path)
    df = pd.read_csv(csv_path, dtype=dtype_dict)
    df = df.replace([np.inf, -np.inf], np.nan)
    y, benign_label, malicious_labels, label_map = infer_labels(
        df,
        category_col=category_col,
        task="multi"
    )
    if benign_label is None:
        raise ValueError("Cannot run targeted-to-non-malicious attack without benign_label.")
    X, used_feature_cols = prepare_feature_matrix(df, feature_cols)
    num_classes = len(np.unique(y))
    print("Feature matrix shape:", X.shape)
    print("Number of used features:", len(used_feature_cols))
    print("Target benign_label:", benign_label)
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed
    )
    all_results = []
    for fold_id, (train_index, test_index) in enumerate(skf.split(X, y), start=1):
        print(f"\n========== {task_name} | Fold {fold_id} ==========")
        X_train_raw = X[train_index]
        X_test_raw = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)
        X_train_scaled = np.nan_to_num(
            X_train_scaled,
            nan=0.0,
            posinf=1e6,
            neginf=-1e6
        ).astype(np.float32)
        X_test_scaled = np.nan_to_num(
            X_test_scaled,
            nan=0.0,
            posinf=1e6,
            neginf=-1e6
        ).astype(np.float32)
        models = build_models(num_classes=num_classes, seed=seed)
        for model_name, model in models.items():
            print(f"\n--- Targeted task: {task_name} | Model: {model_name} ---")
            model_attack_params = {}
            if attack_params is not None:
                model_attack_params = attack_params.get(model_name, {})
            if model_attack_params:
                print("Using attack params:", model_attack_params)
            model.fit(X_train_scaled, y_train)
            pred_clean = model.predict(X_test_scaled).astype(int)
            clean_acc = accuracy_score(y_test, pred_clean)
            clean_f1 = safe_weighted_f1(y_test, pred_clean)
            targetable_mask = np.isin(y_test, malicious_labels) & (pred_clean == y_test)
            attack_indices = np.where(targetable_mask)[0]
            print("Targetable samples:", len(attack_indices))
            snn = train_surrogate_snn(
                X_train=X_train_scaled,
                target_model=model,
                num_classes=num_classes,
                epochs=snn_epochs,
                batch_size=256,
                lr=1e-3
            )
            agreement = surrogate_agreement(
                snn=snn,
                target_model=model,
                X=X_test_scaled
            )
            print(f"SNN Agreement: {agreement:.4f}")
            for eps in epsilons:
                print(f"\nTargeted epsilon = {eps:.2f}")
                rows = []
                fgsm_params = model_attack_params.get("FGSM-SNN", {})
                fgsm_epsilon = eps
                print(f"Running targeted FGSM-SNN -> benign, epsilon={fgsm_epsilon:.2f}")
                X_fgsm = fgsm_snn_targeted_attack(
                    snn=snn,
                    X=X_test_scaled,
                    target_label=benign_label,
                    epsilon=fgsm_epsilon,
                    attack_indices=attack_indices
                )
                pred_fgsm = model.predict(X_fgsm).astype(int)
                rows.append(
                    build_targeted_non_malicious_row(
                        task_name,
                        fold_id,
                        model_name,
                        "FGSM-SNN-targeted-benign",
                        fgsm_epsilon,
                        {},
                        y_test,
                        pred_clean,
                        pred_fgsm,
                        clean_acc,
                        clean_f1,
                        agreement,
                        malicious_labels,
                        benign_label,
                        num_classes,
                        len(used_feature_cols),
                        len(attack_indices)
                    )
                )
                for pgd_steps in pgd_steps_list:
                    pgd_key = f"PGD-{pgd_steps}-SNN"
                    pgd_params = model_attack_params.get(pgd_key, {})
                    pgd_epsilon = eps
                    pgd_steps_for_attack = int(pgd_params.get("steps", pgd_steps))
                    alpha_factor = pgd_params.get("alpha_factor", 5.0)
                    alpha = pgd_params.get("alpha", pgd_epsilon / alpha_factor)
                    print(
                        f"Running targeted PGD-{pgd_steps_for_attack}-SNN -> benign, "
                        f"epsilon={pgd_epsilon:.2f}, alpha={alpha:.6f}"
                    )
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                    X_pgd = pgd_snn_targeted_attack(
                        snn=snn,
                        X=X_test_scaled,
                        target_label=benign_label,
                        epsilon=pgd_epsilon,
                        alpha=alpha,
                        steps=pgd_steps_for_attack,
                        attack_indices=attack_indices
                    )
                    pred_pgd = model.predict(X_pgd).astype(int)
                    rows.append(
                        build_targeted_non_malicious_row(
                            task_name,
                            fold_id,
                            model_name,
                            f"PGD-{pgd_steps_for_attack}-SNN-targeted-benign",
                            pgd_epsilon,
                            {
                                "steps": pgd_steps_for_attack,
                                "alpha_factor": alpha_factor,
                                "alpha": alpha
                            },
                            y_test,
                            pred_clean,
                            pred_pgd,
                            clean_acc,
                            clean_f1,
                            agreement,
                            malicious_labels,
                            benign_label,
                            num_classes,
                            len(used_feature_cols),
                            len(attack_indices)
                        )
                    )
                bba_params = model_attack_params.get("BBA", {})
                bba_epsilon = eps
                bba_max_attempts = bba_params.get("max_attempts", 50)
                bba_top_k = bba_params.get("top_k", 90)
                bba_perturb_features = bba_params.get("perturb_features", 80)
                print(
                    f"Running targeted BBA -> benign, epsilon={bba_epsilon:.2f}, "
                    f"max_attempts={bba_max_attempts}, top_k={bba_top_k}, "
                    f"perturb_features={bba_perturb_features}"
                )
                X_bba = bba_targeted_attack(
                    model=model,
                    X=X_test_scaled,
                    target_label=benign_label,
                    epsilon=bba_epsilon,
                    attack_indices=attack_indices,
                    max_attempts=bba_max_attempts,
                    top_k=bba_top_k,
                    perturb_features=bba_perturb_features,
                    random_state=seed,
                    n_jobs=N_JOBS
                )
                pred_bba = model.predict(X_bba).astype(int)
                rows.append(
                    build_targeted_non_malicious_row(
                        task_name,
                        fold_id,
                        model_name,
                        "BBA-targeted-benign",
                        bba_epsilon,
                        {
                            "max_attempts": bba_max_attempts,
                            "top_k": bba_top_k,
                            "perturb_features": bba_perturb_features
                        },
                        y_test,
                        pred_clean,
                        pred_bba,
                        clean_acc,
                        clean_f1,
                        agreement,
                        malicious_labels,
                        benign_label,
                        num_classes,
                        len(used_feature_cols),
                        len(attack_indices)
                    )
                )
                dba_params = model_attack_params.get("DBA-style", {})
                dba_epsilon = eps
                dba_max_features = dba_params.get("max_features", 5)
                print(
                    f"Running targeted DBA-style -> benign, epsilon={dba_epsilon:.2f}, "
                    f"max_features={dba_max_features}"
                )
                X_dba = dba_style_targeted_attack(
                    model=model,
                    X=X_test_scaled,
                    target_label=benign_label,
                    epsilon=dba_epsilon,
                    attack_indices=attack_indices,
                    max_features=dba_max_features,
                    n_jobs=N_JOBS
                )
                pred_dba = model.predict(X_dba).astype(int)
                rows.append(
                    build_targeted_non_malicious_row(
                        task_name,
                        fold_id,
                        model_name,
                        "DBA-style-targeted-benign",
                        dba_epsilon,
                        {"max_features": dba_max_features},
                        y_test,
                        pred_clean,
                        pred_dba,
                        clean_acc,
                        clean_f1,
                        agreement,
                        malicious_labels,
                        benign_label,
                        num_classes,
                        len(used_feature_cols),
                        len(attack_indices)
                    )
                )
                all_results.extend(rows)
                del rows
                del X_fgsm, pred_fgsm
                del X_pgd, pred_pgd
                del X_bba, pred_bba
                del X_dba, pred_dba
                clear_runtime_cache()
    results_df = pd.DataFrame(all_results)
    result_file = f"{output_prefix}_multi_targeted_non_malicious_fold_results.csv"
    summary_file = f"{output_prefix}_multi_targeted_non_malicious_summary.csv"
    results_df.to_csv(result_file, index=False)
    summary = results_df.groupby(
        ["task", "model", "attack", "epsilon"],
        as_index=False
    ).agg({
        "clean_acc": "mean",
        "clean_f1": "mean",
        "adv_acc": "mean",
        "adv_f1": "mean",
        "acc_drop": "mean",
        "targeted_evasion_asr": "mean",
        "snn_agreement": "mean",
        "num_features": "mean",
        "num_targetable": "mean",
        "target_label": "first"
    })
    summary.to_csv(summary_file, index=False)
    print("\n===== Targeted Non-Malicious Summary =====")
    print(summary)
    print("\nSaved:")
    print(result_file)
    print(summary_file)
    return results_df, summary

def run_all_attack_experiments(
    binary_csv,
    multi_csv,
    dtype_dict,
    feature_cols,
    epsilons=(0.05, 0.10, 0.20, 0.30),
    pgd_steps_list=(10, 20),
    seed=10,
    n_splits=4,
    snn_epochs=80,
    output_prefix="attack_results",
    attack_params=None
):
    binary_results, binary_summary = run_attack_experiment_for_task(
        csv_path=binary_csv,
        dtype_dict=dtype_dict,
        feature_cols=feature_cols,
        task_name="binary",
        category_col="category",
        epsilons=epsilons,
        pgd_steps_list=pgd_steps_list,
        seed=seed,
        n_splits=n_splits,
        snn_epochs=snn_epochs,
        output_prefix=output_prefix,
        attack_params=attack_params
    )
    multi_results, multi_summary = run_attack_experiment_for_task(
        csv_path=multi_csv,
        dtype_dict=dtype_dict,
        feature_cols=feature_cols,
        task_name="multi",
        category_col="category",
        epsilons=epsilons,
        pgd_steps_list=pgd_steps_list,
        seed=seed,
        n_splits=n_splits,
        snn_epochs=snn_epochs,
        output_prefix=output_prefix,
        attack_params=attack_params
    )
    combined_results = pd.concat([binary_results, multi_results], ignore_index=True)
    combined_summary = pd.concat([binary_summary, multi_summary], ignore_index=True)
    combined_results_file = f"{output_prefix}_combined_fold_results.csv"
    combined_summary_file = f"{output_prefix}_combined_summary.csv"
    combined_results.to_csv(combined_results_file, index=False)
    combined_summary.to_csv(combined_summary_file, index=False)
    print("\n===== Combined Summary =====")
    print(combined_summary)
    print("\nSaved combined files:")
    print(combined_results_file)
    print(combined_summary_file)
    return combined_results, combined_summary

def run_temporal_attack_experiments(
    binary_csv,
    multi_csv,
    dtype_dict,
    time_list,
    epsilons=(0.05, 0.10, 0.20, 0.30),
    pgd_steps_list=(10, 20),
    seed=10,
    n_splits=4,
    snn_epochs=80,
    output_prefix="temporal_all_features_attack",
    attack_params=None
):
    all_temporal_results = []
    all_temporal_summaries = []
    for time_suffix in time_list:
        print("\n" + "#" * 100)
        print(f"Running temporal experiment for time window: {time_suffix}")
        print("#" * 100)
        feature_cols = build_feature_cols_for_time(time_suffix)
        current_output_prefix = f"{output_prefix}_{time_suffix}"
        combined_results, combined_summary = run_all_attack_experiments(
            binary_csv=binary_csv,
            multi_csv=multi_csv,
            dtype_dict=dtype_dict,
            feature_cols=feature_cols,
            epsilons=epsilons,
            pgd_steps_list=pgd_steps_list,
            seed=seed,
            n_splits=n_splits,
            snn_epochs=snn_epochs,
            output_prefix=current_output_prefix,
            attack_params=attack_params
        )
        combined_results["time_window"] = time_suffix
        combined_summary["time_window"] = time_suffix
        all_temporal_results.append(combined_results)
        all_temporal_summaries.append(combined_summary)
    temporal_results = pd.concat(all_temporal_results, ignore_index=True)
    temporal_summary = pd.concat(all_temporal_summaries, ignore_index=True)
    temporal_results_file = f"{output_prefix}_ALL_TIME_fold_results.csv"
    temporal_summary_file = f"{output_prefix}_ALL_TIME_summary.csv"
    temporal_results.to_csv(temporal_results_file, index=False)
    temporal_summary.to_csv(temporal_summary_file, index=False)
    print("\n" + "=" * 100)
    print("All temporal experiments completed.")
    print("Saved:")
    print(temporal_results_file)
    print(temporal_summary_file)
    print("=" * 100)
    return temporal_results, temporal_summary

def run_temporal_multi_targeted_non_malicious_experiments(
    multi_csv,
    dtype_dict,
    time_list,
    epsilons=(0.05, 0.10, 0.20, 0.30),
    pgd_steps_list=(10, 20),
    seed=10,
    n_splits=4,
    snn_epochs=80,
    output_prefix="temporal_multi_targeted_non_malicious_attack",
    attack_params=None
):
    all_temporal_results = []
    all_temporal_summaries = []
    for time_suffix in time_list:
        print("\n" + "#" * 100)
        print(f"Running targeted non-malicious temporal experiment for: {time_suffix}")
        print("#" * 100)
        feature_cols = build_feature_cols_for_time(time_suffix)
        current_output_prefix = f"{output_prefix}_{time_suffix}"
        results_df, summary_df = run_multi_targeted_non_malicious_attack_experiment(
            csv_path=multi_csv,
            dtype_dict=dtype_dict,
            feature_cols=feature_cols,
            category_col="category",
            epsilons=epsilons,
            pgd_steps_list=pgd_steps_list,
            seed=seed,
            n_splits=n_splits,
            snn_epochs=snn_epochs,
            output_prefix=current_output_prefix,
            attack_params=attack_params
        )
        results_df["time_window"] = time_suffix
        summary_df["time_window"] = time_suffix
        all_temporal_results.append(results_df)
        all_temporal_summaries.append(summary_df)
    temporal_results = pd.concat(all_temporal_results, ignore_index=True)
    temporal_summary = pd.concat(all_temporal_summaries, ignore_index=True)
    temporal_results_file = f"{output_prefix}_ALL_TIME_fold_results.csv"
    temporal_summary_file = f"{output_prefix}_ALL_TIME_summary.csv"
    temporal_results.to_csv(temporal_results_file, index=False)
    temporal_summary.to_csv(temporal_summary_file, index=False)
    print("\n" + "=" * 100)
    print("All targeted non-malicious temporal experiments completed.")
    print("Saved:")
    print(temporal_results_file)
    print(temporal_summary_file)
    print("=" * 100)
    return temporal_results, temporal_summary
def build_dtype_dict_for_feature_cols(feature_cols):
    local_dtype_dict = {}
    for f in feature_cols:
        local_dtype_dict[f] = float
    return local_dtype_dict
def build_bba_random_grid(seed=10, n_samples=200):
    """
    Build a random-search grid for BBA.
    Constraint:
    perturb_features <= top_k
    """
    rng = np.random.default_rng(seed)
    max_attempts_choices = list(range(10, 201, 10))
    top_k_choices = list(range(20, 301, 10))
    perturb_features_choices = list(range(5, 201, 5))
    candidate_pool = [
        {
            "max_attempts": max_attempts,
            "top_k": top_k,
            "perturb_features": perturb_features
        }
        for max_attempts, top_k, perturb_features in itertools.product(
            max_attempts_choices,
            top_k_choices,
            perturb_features_choices
        )
        if perturb_features <= top_k
    ]
    n_samples = min(n_samples, len(candidate_pool))
    selected_indices = rng.choice(
        len(candidate_pool),
        size=n_samples,
        replace=False
    )
    return [candidate_pool[i] for i in selected_indices]

def evaluate_attack_result(
    task_name,
    model_name,
    attack_name,
    params,
    y_test,
    pred_clean,
    pred_adv,
    clean_acc,
    clean_f1,
    agreement,
    malicious_labels,
    benign_label
):
    adv_acc = accuracy_score(y_test, pred_adv)
    adv_f1 = safe_weighted_f1(y_test, pred_adv)
    asr = attack_success_rate(
        y_true=y_test,
        pred_clean=pred_clean,
        pred_adv=pred_adv,
        malicious_labels=malicious_labels
    )
    evasion_asr = evasion_success_rate(
        y_true=y_test,
        pred_clean=pred_clean,
        pred_adv=pred_adv,
        malicious_labels=malicious_labels,
        benign_label=benign_label
    )
    row = {
        "task": task_name,
        "model": model_name,
        "attack": attack_name,
        "params": str(params),
        "clean_acc": clean_acc,
        "clean_f1": clean_f1,
        "adv_acc": adv_acc,
        "adv_f1": adv_f1,
        "acc_drop": clean_acc - adv_acc,
        "asr": asr,
        "evasion_asr": evasion_asr,
        "snn_agreement": agreement
    }
    row.update(params)
    return row

def quick_search_attack_params_14600_binary_wide(
    binary_csv="binary.csv",
    time_suffix=" (14600.0days)",
    fixed_epsilon=0.30,
    fixed_pgd_steps=10,
    seed=10,
    test_size=0.25,
    snn_epochs=80,
    bba_random_samples=200,
    output_prefix="pgd10_only_param_search_14600_binary_eps030"
):
    set_seed(seed)
    print("\n" + "=" * 100)
    print("Wide parameter search: binary classification, 14600-day all features")
    print(f"Fixed epsilon = {fixed_epsilon}")
    print(f"Fixed PGD steps = {fixed_pgd_steps}")
    print("Selection criterion: maximum accuracy drop")
    print("=" * 100)
    transaction_feature_cols = [
        each_feature + time_suffix
        for each_feature in without_time_list
    ]
    code_feature_cols = list(dict.fromkeys(code_feature_list))
    feature_cols = transaction_feature_cols + code_feature_cols
    feature_cols = list(dict.fromkeys(feature_cols))
    local_dtype_dict = build_dtype_dict_for_feature_cols(feature_cols)
    print("Loading dataset:", binary_csv)
    print("Time suffix:", time_suffix)
    print("Transaction features:", len(transaction_feature_cols))
    print("Code features:", len(code_feature_cols))
    print("Total expected features:", len(feature_cols))
    print("\nFirst 5 transaction features:")
    for f in transaction_feature_cols[:5]:
        print("  ", f)
    print("\nFirst 5 code features:")
    for f in code_feature_cols[:5]:
        print("  ", f)
    df = pd.read_csv(binary_csv, dtype=local_dtype_dict)
    df = df.replace([np.inf, -np.inf], np.nan)
    y, benign_label, malicious_labels, label_map = infer_labels(
        df,
        category_col="category",
        task="binary"
    )
    X, used_feature_cols = prepare_feature_matrix(df, feature_cols)
    print("\nFeature matrix shape:", X.shape)
    print("Number of actually used features:", len(used_feature_cols))
    print("Malicious labels:", malicious_labels)
    print("Benign label:", benign_label)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    X_train_scaled = np.nan_to_num(
        X_train_scaled,
        nan=0.0,
        posinf=1e6,
        neginf=-1e6
    ).astype(np.float32)
    X_test_scaled = np.nan_to_num(
        X_test_scaled,
        nan=0.0,
        posinf=1e6,
        neginf=-1e6
    ).astype(np.float32)
    print("X_train finite:", np.isfinite(X_train_scaled).all())
    print("X_test finite:", np.isfinite(X_test_scaled).all())
    num_classes = len(np.unique(y))
    models = build_models(num_classes=num_classes, seed=seed)
    epsilon = fixed_epsilon
    pgd_steps = fixed_pgd_steps
    pgd_alpha_grid = [
        {"alpha_factor": alpha_factor}
        for alpha_factor in np.round(np.arange(2.0, 20.0 + 0.5, 0.5), 2)
    ]
    bba_grid = []
    dba_grid = []
    print("\nSearch configuration:")
    print("FGSM configs: 0 (skipped)")
    print("PGD alpha configs:", len(pgd_alpha_grid), "with fixed steps =", pgd_steps)
    print("PGD alpha_factor range: 2.0 to 20.0, step 0.5")
    print("BBA random configs: 0 (skipped)")
    print("BBA ranges: skipped")
    print("DBA-style configs: 0 (skipped)")
    print("DBA max_features range: skipped")
    all_rows = []
    for model_name, model in models.items():
        print("\n" + "-" * 100)
        print("Training target model:", model_name)
        print("-" * 100)
        model.fit(X_train_scaled, y_train)
        pred_clean = model.predict(X_test_scaled).astype(int)
        clean_acc = accuracy_score(y_test, pred_clean)
        clean_f1 = safe_weighted_f1(y_test, pred_clean)
        print(f"Clean Accuracy: {clean_acc:.4f}")
        print(f"Clean Weighted F1: {clean_f1:.4f}")
        print("Training SNN surrogate...")
        snn = train_surrogate_snn(
            X_train=X_train_scaled,
            target_model=model,
            num_classes=num_classes,
            epochs=snn_epochs,
            batch_size=256,
            lr=1e-3
        )
        agreement = surrogate_agreement(
            snn=snn,
            target_model=model,
            X=X_test_scaled
        )
        print(f"SNN Agreement: {agreement:.4f}")
        for alpha_params in pgd_alpha_grid:
            alpha_factor = alpha_params["alpha_factor"]
            alpha = epsilon / alpha_factor
            params = {
                "epsilon": epsilon,
                "steps": pgd_steps,
                "alpha_factor": alpha_factor,
                "alpha": alpha
            }
            print(f"[{model_name}] PGD-{pgd_steps}-SNN params={params}")
            X_adv = pgd_snn_attack(
                snn=snn,
                X=X_test_scaled,
                y=y_test,
                epsilon=epsilon,
                alpha=alpha,
                steps=pgd_steps
            )
            pred_adv = model.predict(X_adv).astype(int)
            row = evaluate_attack_result(
                task_name="binary",
                model_name=model_name,
                attack_name=f"PGD-{pgd_steps}-SNN",
                params=params,
                y_test=y_test,
                pred_clean=pred_clean,
                pred_adv=pred_adv,
                clean_acc=clean_acc,
                clean_f1=clean_f1,
                agreement=agreement,
                malicious_labels=malicious_labels,
                benign_label=benign_label
            )
            all_rows.append(row)
        continue
        fgsm_params = {
            "epsilon": epsilon
        }
        print(f"[{model_name}] FGSM-SNN params={fgsm_params}")
        X_adv = fgsm_snn_attack(
            snn=snn,
            X=X_test_scaled,
            y=y_test,
            epsilon=epsilon
        )
        pred_adv = model.predict(X_adv).astype(int)
        row = evaluate_attack_result(
            task_name="binary",
            model_name=model_name,
            attack_name="FGSM-SNN",
            params=fgsm_params,
            y_test=y_test,
            pred_clean=pred_clean,
            pred_adv=pred_adv,
            clean_acc=clean_acc,
            clean_f1=clean_f1,
            agreement=agreement,
            malicious_labels=malicious_labels,
            benign_label=benign_label
        )
        all_rows.append(row)
        for alpha_params in pgd_alpha_grid:
            alpha_factor = alpha_params["alpha_factor"]
            alpha = epsilon / alpha_factor
            params = {
                "epsilon": epsilon,
                "steps": pgd_steps,
                "alpha_factor": alpha_factor,
                "alpha": alpha
            }
            print(f"[{model_name}] PGD-{pgd_steps}-SNN params={params}")
            X_adv = pgd_snn_attack(
                snn=snn,
                X=X_test_scaled,
                y=y_test,
                epsilon=epsilon,
                alpha=alpha,
                steps=pgd_steps
            )
            pred_adv = model.predict(X_adv).astype(int)
            row = evaluate_attack_result(
                task_name="binary",
                model_name=model_name,
                attack_name=f"PGD-{pgd_steps}-SNN",
                params=params,
                y_test=y_test,
                pred_clean=pred_clean,
                pred_adv=pred_adv,
                clean_acc=clean_acc,
                clean_f1=clean_f1,
                agreement=agreement,
                malicious_labels=malicious_labels,
                benign_label=benign_label
            )
            all_rows.append(row)
        for bba_params in bba_grid:
            params = {
                "epsilon": epsilon,
                "max_attempts": bba_params["max_attempts"],
                "top_k": bba_params["top_k"],
                "perturb_features": bba_params["perturb_features"]
            }
            print(f"[{model_name}] BBA params={params}")
            X_adv = bba_attack(
                model=model,
                X=X_test_scaled,
                y=y_test,
                epsilon=epsilon,
                max_attempts=bba_params["max_attempts"],
                top_k=bba_params["top_k"],
                perturb_features=bba_params["perturb_features"],
                random_state=seed,
                n_jobs=N_JOBS
            )
            pred_adv = model.predict(X_adv).astype(int)
            row = evaluate_attack_result(
                task_name="binary",
                model_name=model_name,
                attack_name="BBA",
                params=params,
                y_test=y_test,
                pred_clean=pred_clean,
                pred_adv=pred_adv,
                clean_acc=clean_acc,
                clean_f1=clean_f1,
                agreement=agreement,
                malicious_labels=malicious_labels,
                benign_label=benign_label
            )
            all_rows.append(row)
        for dba_params in dba_grid:
            params = {
                "epsilon": epsilon,
                "max_features": dba_params["max_features"]
            }
            print(f"[{model_name}] DBA-style params={params}")
            X_adv = dba_style_attack(
                model=model,
                X=X_test_scaled,
                y=y_test,
                epsilon=epsilon,
                max_features=dba_params["max_features"],
                n_jobs=N_JOBS
            )
            pred_adv = model.predict(X_adv).astype(int)
            row = evaluate_attack_result(
                task_name="binary",
                model_name=model_name,
                attack_name="DBA-style",
                params=params,
                y_test=y_test,
                pred_clean=pred_clean,
                pred_adv=pred_adv,
                clean_acc=clean_acc,
                clean_f1=clean_f1,
                agreement=agreement,
                malicious_labels=malicious_labels,
                benign_label=benign_label
            )
            all_rows.append(row)
    results_df = pd.DataFrame(all_rows)
    raw_file = f"{output_prefix}_raw.csv"
    results_df.to_csv(raw_file, index=False)
    best_by_model_attack = (
        results_df
        .sort_values(
            by=["model", "attack", "acc_drop", "asr", "adv_acc"],
            ascending=[True, True, False, False, True]
        )
        .groupby(["model", "attack"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best_by_model_attack_file = f"{output_prefix}_best_by_model_attack.csv"
    best_by_model_attack.to_csv(best_by_model_attack_file, index=False)
    best_by_model = (
        results_df
        .sort_values(
            by=["model", "acc_drop", "asr", "adv_acc"],
            ascending=[True, False, False, True]
        )
        .groupby(["model"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best_by_model_file = f"{output_prefix}_best_by_model.csv"
    best_by_model.to_csv(best_by_model_file, index=False)
    ranking_df = results_df.sort_values(
        by=["acc_drop", "asr", "adv_acc"],
        ascending=[False, False, True]
    ).reset_index(drop=True)
    ranking_file = f"{output_prefix}_ranking.csv"
    ranking_df.to_csv(ranking_file, index=False)
    best_overall = ranking_df.head(1).copy()
    best_overall_file = f"{output_prefix}_best_overall.csv"
    best_overall.to_csv(best_overall_file, index=False)
    print("\n" + "=" * 100)
    print("Wide parameter search completed.")
    print("Selection criterion: maximum accuracy drop")
    print("Saved:")
    print(raw_file)
    print(best_by_model_attack_file)
    print(best_by_model_file)
    print(ranking_file)
    print(best_overall_file)
    print("=" * 100)
    print("\nBest parameters by model and attack based on maximum accuracy drop:")
    print(best_by_model_attack[[
        "model",
        "attack",
        "params",
        "clean_acc",
        "adv_acc",
        "acc_drop",
        "asr",
        "evasion_asr",
        "snn_agreement"
    ]])
    print("\nBest result by model based on maximum accuracy drop:")
    print(best_by_model[[
        "model",
        "attack",
        "params",
        "clean_acc",
        "adv_acc",
        "acc_drop",
        "asr",
        "evasion_asr",
        "snn_agreement"
    ]])
    print("\nBest overall result based on maximum accuracy drop:")
    print(best_overall[[
        "model",
        "attack",
        "params",
        "clean_acc",
        "adv_acc",
        "acc_drop",
        "asr",
        "evasion_asr",
        "snn_agreement"
    ]])
    return results_df, best_by_model_attack, best_by_model, ranking_df, best_overall

# ============================================================
# Experiment entry points
# ============================================================
# Usage for other users:
# 1. Edit EXPERIMENT_TO_RUN below, then run: python experiments.py
# 2. Or keep this file unchanged and run with an environment variable:
#    EXPERIMENT_NAME=temporal_attack_all python experiments.py
# 3. To print the available experiment names:
#    EXPERIMENT_NAME=list python experiments.py
#
# Dataset assumptions:
# - binary_csv and multi_csv are configured near the top of this file.
# - category_col is expected to be "category".
# - EXPERIMENT_TIME_SUFFIX must match the transaction feature suffix in the CSV.
#
# Attack coverage in the attack experiments:
# - FGSM-SNN
# - PGD-10-SNN / PGD-20-SNN depending on pgd_steps_list
# - BBA
# - DBA-style

# Choose the default experiment here.
# Available names are the keys in EXPERIMENTS below.
EXPERIMENT_TO_RUN = "temporal_defense"

# Used by single-time-window experiments. Change this to any suffix in time_list
# if you want one specific temporal feature window instead of all windows.
EXPERIMENT_TIME_SUFFIX = time_list[-1]


def _experiment_time_label(time_suffix):
    return (
        time_suffix.strip()
        .replace("(", "")
        .replace(")", "")
        .replace(" ", "_")
        .replace(".", "p")
    )


def _single_window_feature_cols():
    return build_feature_cols_for_time(EXPERIMENT_TIME_SUFFIX)


def run_binary_attack_entry():
    # Binary classification attack experiment for one selected time window.
    # Output: fold-level CSV and summary CSV for binary.csv only.
    time_label = _experiment_time_label(EXPERIMENT_TIME_SUFFIX)
    return run_attack_experiment_for_task(
        csv_path=binary_csv,
        dtype_dict=dtype_dict,
        feature_cols=_single_window_feature_cols(),
        task_name="binary",
        category_col="category",
        epsilons=(0.30,),
        pgd_steps_list=(20,),
        seed=10,
        n_splits=4,
        snn_epochs=80,
        output_prefix=f"attack_results_binary_{time_label}",
        attack_params=BEST_ATTACK_PARAMS
    )


def run_multi_attack_entry():
    # Multi-class classification attack experiment for one selected time window.
    # Output: fold-level CSV and summary CSV for multi.csv only.
    time_label = _experiment_time_label(EXPERIMENT_TIME_SUFFIX)
    return run_attack_experiment_for_task(
        csv_path=multi_csv,
        dtype_dict=dtype_dict,
        feature_cols=_single_window_feature_cols(),
        task_name="multi",
        category_col="category",
        epsilons=(0.30,),
        pgd_steps_list=(20,),
        seed=10,
        n_splits=4,
        snn_epochs=80,
        output_prefix=f"attack_results_multi_{time_label}",
        attack_params=BEST_ATTACK_PARAMS
    )


def run_all_attack_entry():
    # Binary + multi-class attack experiments for one selected time window.
    # Output: binary, multi, and combined result/summary CSV files.
    time_label = _experiment_time_label(EXPERIMENT_TIME_SUFFIX)
    return run_all_attack_experiments(
        binary_csv=binary_csv,
        multi_csv=multi_csv,
        dtype_dict=dtype_dict,
        feature_cols=_single_window_feature_cols(),
        epsilons=(0.30,),
        pgd_steps_list=(20,),
        seed=10,
        n_splits=4,
        snn_epochs=80,
        output_prefix=f"attack_results_all_{time_label}",
        attack_params=BEST_ATTACK_PARAMS
    )


def run_temporal_attack_entry():
    # Binary + multi-class attack experiments across every suffix in time_list.
    # Output: per-window CSV files plus ALL_TIME combined result/summary CSV files.
    return run_temporal_attack_experiments(
        binary_csv=binary_csv,
        multi_csv=multi_csv,
        dtype_dict=dtype_dict,
        time_list=time_list,
        epsilons=(0.30,),
        pgd_steps_list=(20,),
        seed=10,
        n_splits=4,
        snn_epochs=80,
        output_prefix="temporal_all_features_attack_eps030",
        attack_params=BEST_ATTACK_PARAMS
    )


def run_multi_targeted_non_malicious_entry():
    # Multi-class targeted attack for one selected time window.
    # Goal: push correctly classified malicious samples toward the benign label.
    time_label = _experiment_time_label(EXPERIMENT_TIME_SUFFIX)
    return run_multi_targeted_non_malicious_attack_experiment(
        csv_path=multi_csv,
        dtype_dict=dtype_dict,
        feature_cols=_single_window_feature_cols(),
        category_col="category",
        epsilons=(0.30,),
        pgd_steps_list=(20,),
        seed=10,
        n_splits=4,
        snn_epochs=80,
        output_prefix=f"multi_targeted_non_malicious_attack_{time_label}",
        attack_params=BEST_ATTACK_PARAMS
    )


def run_temporal_multi_targeted_non_malicious_entry():
    # Multi-class targeted attack across every suffix in time_list.
    # Goal: compare targeted-to-benign performance over temporal windows.
    return run_temporal_multi_targeted_non_malicious_experiments(
        multi_csv=multi_csv,
        dtype_dict=dtype_dict,
        time_list=time_list,
        epsilons=(0.30,),
        pgd_steps_list=(20,),
        seed=10,
        n_splits=4,
        snn_epochs=80,
        output_prefix="temporal_multi_targeted_non_malicious_attack_eps030",
        attack_params=BEST_ATTACK_PARAMS
    )


def run_quick_param_search_entry():
    # Wide PGD alpha-factor search for binary.csv with 14600-day features.
    # Output: raw search results, best-by-model/attack, ranking, and best overall CSV files.
    return quick_search_attack_params_14600_binary_wide(
        binary_csv=binary_csv,
        time_suffix=" (14600.0days)",
        fixed_epsilon=0.30,
        fixed_pgd_steps=10,
        seed=10,
        test_size=0.25,
        snn_epochs=80,
        bba_random_samples=200,
        output_prefix="pgd10_only_param_search_14600_binary_eps030"
    )


def run_temporal_defense_entry():
    # Temporal adversarial-training defense experiment.
    # This entry matches the original bottom-of-file default call.
    # It requires run_temporal_defense_experiments to be defined/imported in this script.
    defense_runner = globals().get("run_temporal_defense_experiments")
    if defense_runner is None:
        raise RuntimeError(
            "run_temporal_defense_experiments is not defined in this file. "
            "Choose another EXPERIMENT_TO_RUN or add/import the defense experiment function."
        )
    return defense_runner(
        binary_csv=binary_csv,
        multi_csv=multi_csv,
        dtype_dict=dtype_dict,
        time_list=time_list,
        epsilons=(0.30,),
        pgd_steps_list=(20,),
        seed=10,
        n_splits=4,
        snn_epochs=80,
        output_prefix="temporal_adversarial_training_defense_eps030",
        attack_params=BEST_ATTACK_PARAMS,
        defense_strategies=(
            "PGD-20-SNN",
            "Mixed"
        ),
        train_epsilon=0.30,
        evaluation_modes=(
            "transfer",
        ),
        max_train_adv_samples=None,
        run_binary=True,
        run_multi=False
    )


EXPERIMENTS = {
    # Single time window, binary.csv only.
    "binary_attack": (
        run_binary_attack_entry,
        "Binary attack experiment for EXPERIMENT_TIME_SUFFIX."
    ),
    # Single time window, multi.csv only.
    "multi_attack": (
        run_multi_attack_entry,
        "Multi-class attack experiment for EXPERIMENT_TIME_SUFFIX."
    ),
    # Single time window, binary.csv + multi.csv.
    "all_attack": (
        run_all_attack_entry,
        "Binary and multi-class attack experiments for EXPERIMENT_TIME_SUFFIX."
    ),
    # All time windows, binary.csv + multi.csv.
    "temporal_attack_all": (
        run_temporal_attack_entry,
        "Binary and multi-class attack experiments for every time suffix."
    ),
    # Single time window, targeted multi-class attack.
    "multi_targeted_non_malicious": (
        run_multi_targeted_non_malicious_entry,
        "Target malicious multi-class samples toward the benign label for one time suffix."
    ),
    # All time windows, targeted multi-class attack.
    "temporal_multi_targeted_non_malicious": (
        run_temporal_multi_targeted_non_malicious_entry,
        "Target malicious multi-class samples toward the benign label across all time suffixes."
    ),
    # Parameter search helper for the 14600-day binary setup.
    "quick_param_search_14600_binary": (
        run_quick_param_search_entry,
        "PGD parameter search for binary.csv with 14600-day features."
    ),
    # Defense experiment, if its implementation is present.
    "temporal_defense": (
        run_temporal_defense_entry,
        "Temporal adversarial-training defense experiment."
    ),
}


def list_available_experiments():
    print("Available experiments:")
    for name, (_, description) in EXPERIMENTS.items():
        print(f"  {name}: {description}")


def main():
    print("CPU_COUNT:", CPU_COUNT)
    print("BBA/DBA parallel n_jobs:", N_JOBS)
    print("PyTorch num_threads:", torch.get_num_threads())
    print("PyTorch interop_threads:", torch.get_num_interop_threads())
    selected_experiment = os.environ.get("EXPERIMENT_NAME", EXPERIMENT_TO_RUN)
    if selected_experiment in ("list", "help"):
        list_available_experiments()
        return None
    if selected_experiment not in EXPERIMENTS:
        list_available_experiments()
        raise ValueError(f"Unknown experiment: {selected_experiment}")
    runner, description = EXPERIMENTS[selected_experiment]
    print("Selected experiment:", selected_experiment)
    print("Description:", description)
    return runner()


if __name__ == "__main__":
    main()
