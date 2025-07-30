################################################################################
# Copyright 2024 alpine-software.at                                            #
################################################################################
# This is free software you can redistribute/modify under the terms of the     #
# GNU Lesser General Public License 3 or later: http://www.gnu.org/licenses    #
################################################################################

import os
import shutil
import itertools
import numpy as np
import pandas as pd
from datetime import datetime

class PROParser:
    _station_parameters_lookup = "[STATION_PARAMETERS]"
    _header_lookup = "[HEADER]"
    _data_lookup = "[DATA]"

    class DataList:
        """Work with PRO files Data parts on a line-by-line basis:
        
            - find indices of dates
            - find indices of codes based on start indices
            - read data based on code and start_index
            - insert line at index
        
        Best suited if one needs to work on lines at the latest dates (at least regarding find_date()). 
        For making changes to every day of pro files containing entire seasons, think about a different approach (using _df?!).

        Examples:
            >>> idx = pro.list_data.find_date("2024.01.01 10:00:00")
            >>> deposition_dates = pro.list_data.read_data("0505", idx)
            >>> pro.list_data.insert_line(idx+len(pro.header)-1, code='1101', data=np.arange(10))
        """
        def __init__(self):
            self.data = []
            self._first_added_data_line_index = None

        def find_date(self, date: str|datetime, throw=False):
            """Searches from DataList end toward beginning (for exact datetime matches!). 
            --> best suited to find index of recent dates"""
            if isinstance(date, datetime):
                date = date.strftime("%d.%m.%Y %H:%M:%S")
            
            for index in range(len(self.data)-1, -1, -1):
                if self.data[index].startswith(f"0500,{date}"):
                    return index
            
            msg = f"Cannot find date {date}"
            if throw:
                raise ValueError(msg)
            else:
                print(f"[w]  {msg}")
            return None
        
        def find_line(self, code: str, start_index=0):
            """Searches in DataList.data from start_index towards end and returns first line with given code."""
            for index in range(start_index, len(self.data)):
                if self.data[index].startswith(code):
                    return index
            return None
        
        def read_data(self, code: str, start_index=0):
            """Searches in DataList.data from start_index towards end and returns first line with given code as numpy array."""
            idx = self.find_line(code, start_index)
            try:
                data = np.array(self.data[idx].split(',')[1:], dtype=float)
                return data
            except ValueError:
                return np.array(self.data[idx].split(',')[1:], dtype=str)

        def insert_line(self, index: int, code: str=None, data: np.ndarray=None, new_line: str=None):
            """Insert a line into self.data at the specified index.
            
            Parameters:
                index:      use find_date() or find_line()
                code:       PRO file code to prepend to data (not needed if new_line is provided)
                data:       array of data to be inserted (not needed if new_line is provided)
                new_line:   str representation that will be inserted as is 
                            (will be assembled automatically if code and data are given)
            """
            if index < 0:
                index = len(self.data) + index  # Convert negative index to positive
            if self._first_added_data_line_index is None:
                self._first_added_data_line_index = index
            else:
                self._first_added_data_line_index = min(index, self._first_added_data_line_index)
            if not new_line:
                formatter = {
                    'float_kind': lambda x: f'{x:g}',
                    'int_kind': lambda x: f'{x:g}',
                    'str_kind': lambda x: '' if x == ' ' else f'{x}'
                }
                new_line = f"{code},{np.array2string(data, formatter=formatter, separator=',', max_line_width=np.inf)[1:-1]}"
            self.data.insert(index, new_line)

    def __init__(self, profile: str):
        self.station_parameters = {}
        self.header = {}
        self._datadict = None
        self.list_data = PROParser.DataList()
        self.profile = None
        self._dict = None
        self.parse_pro(profile)

    def parse_pro(self, profile: str):
        with open(profile, "r") as fpro:
            station_parameters_part = False
            header_part = False
            data_part = False
            for line_index, line in enumerate(fpro):
                line = line.partition("#")[0].strip() # before comments
                if not line:
                    continue
                if line == PROParser._station_parameters_lookup:
                    station_parameters_part = True
                    continue
                elif line == PROParser._header_lookup:
                    header_part = True
                    continue
                elif line == PROParser._data_lookup:
                    data_part = True
                    self._data_lookup_index = line_index
                    continue

                if data_part:
                    self.list_data.data.append(line)
                elif header_part:
                    # TODO: code 0530 is not caught properly (report bug to SNOWPACK?!)
                    fields = [field.strip() for field in line.split(",")]
                    self.header[fields[0]] = fields[1:]
                elif station_parameters_part:
                    fields = [field.strip() for field in line.split("=")]
                    self.station_parameters[fields[0]] = fields[1]
        self.profile = profile
        self._dict = None

    def _df(self):
        df = pd.DataFrame(
            [ll.split(",") for ll in self.list_data]
        )
        return df

    def _dev_to_dict(self):
        """Quick parsing of pro-file codes. Aimed to provide the
        most comprehensive containers but is performing extremely poorly -
        only call on hours' worth of data! Use the appropriate
        packages for operational use and full datasets.
        """
        if not self.list_data:
            return None
        if self._dict is None:
            df = self._df()
            date_marker = "0500"
            indices = df[df[0] == date_marker].index
            dregions = {}
            drange = zip(indices, indices[1:])
            for ss, ee in drange:
                region = df.iloc[ss + 1:ee]
                dregions[df.loc[ss, 1]] = region
                
            self._datadict = {}
            for dt in dregions:
                day = dregions[dt]
                day.set_index(0, inplace=True)
                day = day.drop(1, axis=1) # drop number of elements
                dd = day.to_dict(orient="index")
                # remove None-values and convert to float:
                dd = {key: {k: float(v) for k, v in val.items() if v} for key, val in dd.items()}
                ll = {key: list(val.values()) for key, val in dd.items()} # un-nest inner dict
                self._datadict[dt] = ll
        return self._datadict

    def write_pro(self, output_file: str=None, write_from_first_added_data_line: bool=True):
        """
        Write the .pro file from the .list_data view (!!), either to a new file or overwrite the existing one.

        This method writes the content of the list_data (type DataList) to the specified output file. If `output_file` is not provided,
        it will overwrite the original file used to create the parser. By default, the method will write from the first 
        newly added data line to the end of the file if `write_from_first_added_data_line` is set to True. Otherwise, 
        it will write the entire content.

        Parameters
        ----------
        output_file : str, optional
            The path to the output file. If None, the original file will be overwritten.
        write_from_first_added_data_line : bool, default=True
            If True and `output_file` is None, only the lines starting from the first newly added data line will be 
            written to the file. If False or a new file is specified, the entire content will be written. (Actually,
            only marginal performance difference..)

        Notes
        -----
        - A temporary file is created during the writing process. If no error occurs, the original file is replaced by 
        this temporary file. In case of an error, the temporary file is removed.
        """
        if output_file is None:
            output_file = self.profile
            overwrite = True
        else:
            overwrite = False

        tmp_file_path = output_file + '.tmp'
        try:
            with open(tmp_file_path, 'w') as tmp_file:
                if overwrite and write_from_first_added_data_line:  # copy first part over and rewrite only from a specific index in data_part
                    with open(output_file, 'r') as original_file:
                        # Copy unchanged lines from the original file to the tmp file
                        copy_up_to_here = self._data_lookup_index + self.list_data._first_added_data_line_index + 1
                        lines_to_copy = list(itertools.islice(original_file, copy_up_to_here))
                        tmp_file.writelines(lines_to_copy)
                        
                        # Write new lines from self.list_data
                        new_lines = self.list_data.data[self.list_data._first_added_data_line_index:]
                        tmp_file.writelines(line + '\n' for line in new_lines)

                else:  # Rewrite every line of the file
                    # Write station parameters
                    tmp_file.write(PROParser._station_parameters_lookup + '\n')
                    tmp_file.writelines(f"{key}={value}\n" for key, value in self.station_parameters.items())

                    # Write header
                    tmp_file.write('\n' + PROParser._header_lookup + '\n')
                    tmp_file.writelines(f"{key}= {','.join(value)}\n" for key, value in self.header.items())

                    # Write data
                    tmp_file.write('\n' + PROParser._data_lookup + '\n')
                    tmp_file.writelines(line + '\n' for line in self.list_data.data)

            shutil.move(tmp_file_path, output_file)
        except Exception as e:
            os.remove(tmp_file_path)
            raise e