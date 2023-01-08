import pandas as pd
import json
from matplotlib import pyplot as plt
import pandasql as ps
csv_filepath_fl = r"C:\Egyetem\Diplomamunka\data\fl_st.csv"
csv_filepath_orig_lab_st = r"C:\Egyetem\Diplomamunka\data\orig_lab_st.csv"
csv_filepath_lab_st = r"C:\Egyetem\Diplomamunka\data\lab_st.csv"
class GetData():
    def __init__(self, path):
        self.path = path
    def read_flame_data(self, nrows_to_read):
        self.df = pd.read_csv(self.path,nrows=nrows_to_read)

        # df_as_json = df.to_json()
        # with open(path.replace(".csv",".json"),"w",) as file:
        #     json.dump(df_as_json, file)
        # print(df_as_json)
    def generate_join_id(self,id_name, columns):
        self.df = self.df.astype(str)
        self.df[id_name] = self.df[columns[0]] + "_" + self.df[columns[1]]
        return self.df
    @staticmethod
    def collect_data_features(df,*column_names):
        for column_name in column_names:
            number_of_appearance = df[column_name].value_counts()
            print("number_of_appearance: "+number_of_appearance)


    def separate(self,separation_number,equal_pieces):
        number = separation_number
        number_minus_one = 0
        generated_files = []
        for i in range(equal_pieces):
            iteration_path = self.path.replace(".csv",f"_{i}.csv")

            self.df.iloc[number_minus_one:number,::].to_csv(iteration_path)
            generated_files.append(iteration_path)
            number_minus_one = number
            number += int(separation_number/equal_pieces)
        return generated_files
    def TODO_implement_a_yield_for_partional_dfs(self):
        pass





def downstream(csv_path, read_inrows):
    df = pd.read_csv(csv_path,nrows=read_inrows)
    #number_of_appearance = df["ROI"].value_counts()
    save_i = []
    for i in range(1, read_inrows):
        x_item = df.iloc[i,1]
        y_item = df.iloc[i,8]
        if x_item < df.iloc[i-1,1]:
            save_i.append(i)

    frame= df["frame"]
    fl = df["fl_norm_spike"]
    print(frame.tail(10))
    print(fl)
    plt.ylabel("Intensity")
    plt.xlabel("Timestamp")
    plt.title("Calcium Imaging EDA")
    begin_row = 0
    for row in save_i:
        plt.plot(frame[begin_row:row:],fl[begin_row:row:])
        begin_row = row
    plt.savefig(r"C:\Egyetem\Diplomamunka\data\sample2.jpg")
    plt.show()

    # df.plot(x=df["frame"], y=df["fl_mean"], kind="line")
    # print(number_of_appearance)

def merge_with_lab(first_df, sec_df, column_first, column_sec):
    res = first_df.merge(sec_df, how='inner', left_on=[column_first], right_on=[column_sec])
    final_df = pd.merge(first_df, sec_df[[column_sec,"drug1_ID","drug2_ID"]], how="left", left_on=[column_first],
                        right_on=[column_sec])

    return final_df
csv_path = r"C:\Egyetem\Diplomamunka\data\fl_st_0.csv"

class QueryDF():

    def __init__(self, df):
        self.df = df

    def query(self,query_statement, environment):
        if environment =="locals":
            q_result = ps.sqldf(query_statement, locals())
        else:
            q_result = ps.sqldf(query_statement, globals())
        return q_result



if __name__ == "__main__":

    # Getter = GetData(csv_filepath_fl)
    # Getter.read_flame_data(1000000)
    # generated_files = Getter.separate(100000,10)

    generated_files = [r"C:\Egyetem\Diplomamunka\data\fl_st_0.csv"]
    flame_df = pd.read_csv(generated_files[0])
    lab_df = pd.read_csv(csv_filepath_lab_st)
    q1 = """SELECT frame,ROI,period,fl_mean,drug1_ID,drug2_ID,drug1_cc_uM,drug2_cc_uM FROM flame_df
            LEFT JOIN lab_df
            ON flame_df.id = lab_df.lab_id """
    SQL = QueryDF(flame_df)
    query_result = SQL.query(q1,"globals")
    query_result.to_csv(generated_files[0].replace(".csv","_queried.csv"))


