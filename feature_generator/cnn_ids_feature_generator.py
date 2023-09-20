import gc
import pandas as pd
import numpy as np
import typing
from scapy.all import *

from . import abstract_feature_generator

DEFAULT_WINDOW_SIZE = 44
DEFAULT_NUMBER_OF_BYTES = 58
DEFAULT_WINDOW_SLIDE = 1
AVTP_PACKETS_LENGHT = 438

class CNNIDSFeatureGenerator(abstract_feature_generator.AbstractFeatureGenerator):
    def __init__(self, config: typing.Dict):
        self._window_size = config.get('window_size', DEFAULT_WINDOW_SIZE)
        self._number_of_bytes = config.get('number_of_bytes', DEFAULT_NUMBER_OF_BYTES)
        self._window_slide = config.get('window_slide', DEFAULT_WINDOW_SLIDE)
        self._number_of_columns = self._number_of_bytes * 2


    def generate_features(self, paths_dictionary: typing.Dict):
        raw_injected_only_packets = self.__read_raw_packets(paths_dictionary['injected_only_frame_path'])
        injected_only_packets_array = self.__convert_raw_packets(raw_injected_only_packets)

        X = np.empty(shape=(0, self._window_size, self._number_of_columns), dtype='uint8')
        y = np.array([], dtype='uint8')

        for injected_raw_packets_path in paths_dictionary['injected_data_paths']:
            # Load raw packets
            raw_packets = self.__read_raw_packets(injected_raw_packets_path)

            # Convert loaded packets to np array with uint8_t size
            packets_array = self.__convert_raw_packets(raw_packets)

            # Preprocess packets
            preprocessed_packets = self.__preprocess_raw_packets(packets_array, split_into_nibbles=True)

            # Generate labels
            labels = self.__generate_labels(packets_array, injected_only_packets_array)

            # Aggregate features and labels
            aggregated_X, aggregated_y = self.__aggregate_based_on_window_size(preprocessed_packets, labels)

            # Concatenate both indoors injected packets
            X = np.concatenate((X, aggregated_X), axis=0, dtype='uint8')
            y = np.concatenate((y, aggregated_y), axis=0, dtype='uint8')

            np.savez(f"{paths_dictionary['output_path']}/X_Wsize_{self._window_size}_Cols_{self._number_of_columns}_Wslide_{self._window_slide}", X)
            np.savez(f"{paths_dictionary['output_path']}/y_Wsize_{self._window_size}_Cols_{self._number_of_columns}_Wslide_{self._window_slide}", y)


    def __read_raw_packets(self, pcap_filepath):
        raw_packets = rdpcap(pcap_filepath)

        raw_packets_list = []

        for packet in raw_packets:
            if (len(packet) == AVTP_PACKETS_LENGHT):
                raw_packets_list.append(raw(packet))

        return raw_packets_list


    def __convert_raw_packets(self, raw_packets_list):
        converted_packets_list = []

        for raw_packet in raw_packets_list:
            converted_packet = np.frombuffer(raw_packet, dtype='uint8')
            converted_packets_list.append(converted_packet)

        return np.array(converted_packets_list, dtype='uint8')


    def __is_array_in_list_of_arrays(self, array_to_check, list_np_arrays):
        # Reference:
        # https://stackoverflow.com/questions/23979146/check-if-numpy-array-is-in-list-of-numpy-arrays
        is_in_list = np.any(np.all(array_to_check == list_np_arrays, axis=1))

        return is_in_list


    def __generate_labels(self, packets_list, injected_packets):
        labels_list = []

        for packet in packets_list:
            current_label = 0

            if self.__is_array_in_list_of_arrays(packet, injected_packets):
                current_label = 1

            labels_list.append(current_label)

        return labels_list


    def __select_packets_bytes(self, packets_list, first_byte=0, last_byte=58):
        selected_packets = packets_list[:, first_byte:last_byte]

        return np.array(selected_packets, dtype='uint8')


    def __calculate_difference_module(self, selected_packets):
        difference_array = np.diff(selected_packets, axis=0)
        difference_module = np.mod(difference_array, 256)

        return difference_module


    def __split_byte_into_nibbles(self, byte):
        high_nibble = (byte >> 4) & 0xf
        low_nibble = (byte) & 0xf

        return high_nibble, low_nibble


    def __create_nibbles_matrix(self, difference_module):
        nibbles_matrix = []

        # Difference matrix has "n" rows and "p" columns
        for row_index in range(len(difference_module)):
            nibbles_row = []
            for column_index in range(len(difference_module[row_index])):
                hi_ni, low_ni = self.__split_byte_into_nibbles(difference_module[row_index, column_index])

                nibbles_row.append(hi_ni)
                nibbles_row.append(low_ni)

            nibbles_matrix.append(np.array(nibbles_row, dtype='uint8'))

        return np.array(nibbles_matrix, dtype='uint8')


    def __preprocess_raw_packets(self, converted_packets, split_into_nibbles=True):
        # Select first 58 bytes
        selected_packets = self.__select_packets_bytes(converted_packets)

        # Calculate difference and module between rows
        diff_module_packets = self.__calculate_difference_module(selected_packets)

        # Split difference into two nibbles
        if split_into_nibbles:
            diff_module_packets = self.__create_nibbles_matrix(diff_module_packets)

        return diff_module_packets


    def __aggregate_based_on_window_size(self, x_data, y_data):
        # Prepare the list for the transformed data
        X, y = list(), list()

        # Loop of the entire data set
        for i in range(x_data.shape[0]):
            # Compute a new (sliding window) index
            start_ix = i*(self._window_slide)
            end_ix = start_ix + self._window_size - 1 + 1

            # If index is larger than the size of the dataset, we stop
            if end_ix >= x_data.shape[0]:
                break

            # Get a sequence of data for x
            seq_X = x_data[start_ix:end_ix]
            if (self._window_slide == 1):
              # Get only the last element of the sequence for y
              seq_y = y_data[end_ix]
            else:
              # If the sequence contains an attack, the label is considered as attack
              tmp_seq_y = y_data[start_ix:end_ix]
              if 1 in tmp_seq_y:
                seq_y = 1
              else:
                seq_y = 0
            # Append the list with sequencies
            X.append(seq_X)
            y.append(seq_y)

        # Make final arrays
        x_array = np.array(X, dtype='uint8')
        y_array = np.array(y, dtype='uint8')

        return x_array, y_array
