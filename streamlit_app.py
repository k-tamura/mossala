import copy
import pytz
import woothee
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import *
from pandas import DataFrame
from datetime import datetime
import re
import json
import pycountry
from urllib.request import urlopen
import ipaddress
from ipaddress import ip_address
   
def is_private_ip_address(IP: str) -> str:
    return True if (ip_address(IP).is_private) else False

def is_valid_ip_address(sample_str):
    ''' Returns True if given string is a
        valid IP Address, else returns False'''
    result = True
    try:
        ipaddress.ip_network(sample_str)
    except:
        result = False
    return result

def get_country(df):
    ip_address = df['Remote Host']
    if ip_address == 'others':
        return 'others', 'others', np.NaN, np.NaN
    if not is_valid_ip_address(ip_address):
        return 'Unknown', 'Unknown', np.NaN, np.NaN
    if is_private_ip_address(ip_address):
        return 'Local', 'Local', np.NaN, np.NaN
    response = urlopen(f'http://ipinfo.io/{ip_address}/json')
    data = json.load(response)
    country = data['country']
    loc = data['loc'].split(',')
    return pycountry.countries.get(alpha_2=country).name, data['city'], float(loc[0]), float(loc[1])

# 円グラフなどに表示する際の少数の要素（1%以下の割合のもの）を「others」（その他）に変換する関数を定義
# DataFrame用
def replace_df_minors_with_others(df_before, column_name):
    elm_num = 1
    for index, row in df_before.sort_values([column_name], ascending=False).iterrows():
        if (row[column_name] / df_before[column_name].sum()) > 0.02:
            elm_num = elm_num + 1
    
    df_after = df_before.sort_values([column_name], ascending=False).nlargest(elm_num, columns=column_name)
    others = df_before.drop(df_after.index)[column_name].sum()
    if others > 0:
        df_after.loc[len(df_after)] = ['others', others]
    return df_after

# 辞書用
def replace_dict_minors_with_others(dict_before):
    dict_after = {}
    others = 0
    total = sum(dict_before.values())
    for key in dict_before.keys():
        if (dict_before.get(key) / total) > 0.02:
            dict_after[key] = dict_before.get(key)
        else:
            others = others + dict_before.get(key)
    dict_after = {k: v for k, v in sorted(dict_after.items(), reverse=True, key=lambda item: item[1])}
    if others > 0:
        dict_after['others'] = others
    return dict_after

def parse_str(x):
    if x[0] == '"':
        return x[1:-1]
    else:
        return x

def parse_datetime(x):
    if x.startswith("["):
        dt = datetime.strptime(x[1:21], '%d/%b/%Y:%H:%M:%S')
    else:
        dt = datetime.strptime(x[0:20], '%d/%b/%Y:%H:%M:%S')
    return dt

def drow_pie_and_table(df, col_name, title):
    st.markdown('### ' + title)
    grouped_df = DataFrame(df.groupby([col_name]).size().index)
    grouped_df['count'] = df.groupby([col_name]).size().values

    grouped_df['percentage'] = (df.groupby([col_name]).size() / len(df) * 100).values
    col1, col2 = st.columns(2)
    if len(grouped_df[grouped_df[col_name].str.len() > 50]) == 0:
        with col1:
            fig = plt.figure()
            grouped_df_with_others = replace_df_minors_with_others(grouped_df.drop('percentage',axis=1), 'count')
            plt.pie(grouped_df_with_others['count'], labels = grouped_df_with_others[col_name], autopct = '%1.1f%%', startangle = 90)
            fig
        with col2:
            st.write(grouped_df.sort_values(['count'], ascending=False))
    else:
        fig = plt.figure()
        grouped_df_with_others = replace_df_minors_with_others(grouped_df.drop('percentage',axis=1), 'count')
        grouped_df_with_others[col_name].where(grouped_df_with_others[col_name].str.len() < 50, grouped_df_with_others[col_name].str[:50] + '...', inplace=True) 
        plt.pie(grouped_df_with_others['count'], labels = grouped_df_with_others[col_name], autopct = '%1.1f%%', startangle = 90)
        fig

        st.write(grouped_df.sort_values(['count'], ascending=False))

def drow_delayed_access(df, is_RT_sec, sec):
    st.markdown(f'### {sec}秒以上の遅延の発生状況')
    fig = plt.figure(figsize = (15, 5))
    delayed_access = df[['Request', 'Response Time']]
    delayed_access.index = df['Time']
    if not is_RT_sec:
        sec = sec * 1000
    delayed_access = delayed_access.resample('S')['Response Time'].apply(lambda x: (x >= sec).sum())
    delayed_access.index.name = 'Time'
    delayed_access.plot()
    plt.title('Total Delayed Access')
    plt.ylabel('Delayed Access')
    ax = plt.gca()
    ax.yaxis.set_major_locator(MultipleLocator(1))
    fig

    # Consideration when request column is not enclosed by commas, like "GET / HTTP/1.0"
    for i in range(len(df_tmp.columns) - 2):
        if df_tmp[i].isin(ALL_METHODS).all():
            df_tmp[i] = df_tmp[i].astype(str).str.cat([df_tmp[i + 1].astype(str), df_tmp[i + 2].astype(str)], sep=' ')
            df_tmp.drop([i+1, i+2], axis=1, inplace=True)
            df_tmp.columns = range(df_tmp.shape[1])
            break

def define_usecols(df_tmp, default_usecols, default_names):
    for i in range(df_tmp.shape[1]):
        colNumeric = pd.to_numeric(df_tmp[i], errors="coerce")
        isNumeric = pd.to_numeric(df_tmp[i], errors="coerce").notna().all()
        if not 'Remote Host' in default_names and is_valid_ip_address(df_tmp.iloc[0, i]):
            default_usecols.append(i)
            default_names.append('Remote Host')
            continue
        if not 'Time' in default_names and type(df_tmp.iloc[0, i]) == str and df_tmp.iloc[0, i].count(':') >= 2:
            default_usecols.append(i)
            default_names.append('Time')
            continue
        if not 'Request' in default_names and type(df_tmp.iloc[0, i]) == str and 'HTTP/' in df_tmp.iloc[0, i]:
            default_usecols.append(i)
            default_names.append('Request')
            continue
        if not 'Status' in default_names and isNumeric and len(df_tmp) == len(df_tmp[(colNumeric >= 100) & (colNumeric < 600)]):
            default_usecols.append(i)
            default_names.append('Status')
            continue
        if not 'Size' in default_names and colNumeric.median() > 3000:
            default_usecols.append(i)
            default_names.append('Size')
            continue
        if not 'User Agent' in default_names and type(df_tmp.iloc[0, i]) == str and 'Mozilla/' in df_tmp.iloc[0, i]:
            default_usecols.append(i)
            default_names.append('User Agent')
            continue
        if not 'Response Time (s)' in default_names and isNumeric and colNumeric.median() <= 3:
            default_usecols.append(i)
            default_names.append('Response Time (s)')
            continue
        if not 'Response Time (ms)' in default_names and isNumeric and colNumeric.median() > 3:
            default_usecols.append(i)
            default_names.append('Response Time (ms)')
            continue

st.set_page_config(page_title="MOSSALA", page_icon='icon.png')
st.title("Multiple OSS Access Log Analyzer")
st.image('logo.png')

'''
**以下のOSSのような一般的な出力形式のアクセスログを解析できます。**
 
* Apache
* Nginx
* Tomcat
* WildFly
'''

ALL_METHODS = ['GET', 'HEAD', 'POST', 'PUT', 'DELETE', 'CONNECT', 'OPTIONS', 'TRACE', 'LINK', 'UNLINK']

uploaded_file = st.file_uploader("アクセスログをアップロードしてください。")
if uploaded_file is not None:

    df_tmp = pd.read_csv(
        copy.deepcopy(uploaded_file),
        sep=r'\s(?=(?:[^"]*"[^"]*")*[^"]*$)(?![^\[]*\])',
        engine='python',
        na_values='-',
        header=None)

    # Consideration when request column is not enclosed by commas, like "GET / HTTP/1.0"
    for i in range(len(df_tmp.columns) - 2):
        if df_tmp[i].isin(ALL_METHODS).all():
            df_tmp[i] = df_tmp[i].astype(str).str.cat([df_tmp[i + 1].astype(str), df_tmp[i + 2].astype(str)], sep=' ')
            df_tmp.drop([i+1, i+2], axis=1, inplace=True)
            df_tmp.columns = range(df_tmp.shape[1])
            break
    
    default_usecols = []
    default_names = []

    define_usecols(df_tmp, default_usecols, default_names)

    st.markdown('### アクセスログ（先頭5件）')
    st.write(df_tmp.head(5))

    help_txt = '''
        以下のフォーマット文字列を解析可能です。

        | 列名 | フォーマット文字列 | 説明 | 
        |:-----|:-----:|:-----|
        | Remote Host | `%h` | リモートホスト |
        | Time | `%t` | リクエストを受付けた時刻 | 
        | Request | `\"%r\"` | リクエストの最初の行 | 
        | Status | `%>s` | ステータス | 
        | Size | `%b` | レスポンスのバイト数 | 
        | User Agent | `\"%{User-agent}i\"` | リクエストのUser-agentヘッダの内容 | 
        | Response Time (ms) | `%D` | リクエストを処理するのにかかった時間（ミリ秒） |         
        | Response Time (s) | `%T` | リクエストを処理するのにかかった時間（秒） |         
        
        詳細については、各OSSの公式ドキュメントを参照して下さい。Apacheの公式ドキュメントを参照する場合は、[ここ](https://httpd.apache.org/docs/2.4/ja/mod/mod_log_config.html)をクリックして下さい。
        '''

    if len(default_usecols) == 0:
        if len(df_tmp.columns) <= 8:
            default_usecols = [0, 3, 4, 5, 6]
            default_names = ['Remote Host', 'Time', 'Request', 'Status', 'Size']
        elif len(df_tmp.columns) > 8:
            default_usecols = [0, 3, 4, 5, 6, 8]
            default_names = ['Remote Host', 'Time', 'Request', 'Status', 'Size', 'User Agent']

    usecols = st.multiselect(
        '何番目の列を解析の対象にしますか？',
        [i for i in range(0,len(df_tmp.columns))],
        default_usecols)
    names = st.multiselect(
        'これらの列を何を意味しますか？',
        ['Remote Host', 'Time', 'Request', 'Status', 'Size', 'User Agent', 'Response Time (ms)', 'Response Time (s)'],
        default_names, help=help_txt)

    is_analyzable = False
        
    if len(usecols) == 0 or len(names) == 0:
        st.error('解析対象の列が指定されていません。')
    elif len(usecols) != len(names):
        st.error('設定した列の数が一致していません。')
    else:
        if 'Time' in names:
            first_row_of_time = df_tmp.iloc[0, usecols[names.index('Time')]]
        if 'Status' in names:
            all_row_of_status = df_tmp.iloc[:, usecols[names.index('Status')]]

        if 'Time' in names \
            and (not first_row_of_time \
            or type(first_row_of_time) != str):
            st.error('Time列の値が時刻形式ではありません。')
        elif 'Status' in names \
            and (all_row_of_status.dtypes != 'int64'
            or (all_row_of_status < 100).sum() > 0 \
            or (all_row_of_status >= 600).sum() > 0):
            st.error('Status列の値がステータスコードの値ではありません。')
        # TODO 他の列の入力チェックも実装
        else:
            is_analyzable = True

    if is_analyzable and st.button('解析開始'):
        st.balloons()
        deafult_converters = {'Remote Host': parse_str,
                        'Time': parse_datetime,
                        'Response Time': int,
                        'Request': parse_str,
                        'Status': int,
                        'Size': int,
                        'User Agent': parse_str}

        my_bar = st.progress(0)

        # usecols、names、convertersはCSVの列順にソートされていないとエラーになるので全てソート
        tmp_usecols = []
        tmp_names = []
        for i in range(len(df_tmp.columns)):
            if i in usecols:
                idx = usecols.index(i)
                tmp_usecols.append(usecols[idx])
                tmp_names.append(names[idx])
        usecols = tmp_usecols
        names = tmp_names
        converters = {}
        for name in names:
            if name == 'Response Time (ms)':
                converters['Response Time'] = int
                is_RT_sec = False
            elif name == 'Response Time (s)':
                converters['Response Time'] = float
                is_RT_sec = True
            elif name in deafult_converters.keys():
                converters[name] = deafult_converters[name]
        names = [('Response Time' if e.startswith('Response Time') else e) for e in names]

        try:
            df = pd.read_csv(
                uploaded_file,
                sep=r'\s(?=(?:[^"]*"[^"]*")*[^"]*$)(?![^\[]*\])',
                engine='python',
                na_values='-',
                header=None,
                usecols=usecols,
                names=list(names),
                converters=converters)
        except ValueError as e:
            st.error('解析に失敗しました。')
            #st.exception(e)
            raise e
        my_bar.progress(20)

        st.markdown('### アクセスログ（解析対象業のみ抽出、先頭5件）')
        st.write(df.head(5))


        if 'Time' in df.columns:
            min_t = df['Time'].min()
            max_t = df['Time'].max()
            st.markdown(f'''
                * ログの件数　　　　 : {len(df)} 件
                * ログを記録した期間 : {min_t.strftime('%Y/%m/%d %H:%M:%S')} 〜 {max_t.strftime('%Y/%m/%d %H:%M:%S')} 
                * 平均リクエスト数　 : {round(len(df) / (max_t - min_t).total_seconds() * 60, 2)} 件 / 分
            ''')
        else:
            st.markdown(f'''
                * ログの件数　　　　 : {len(df)} 件
            ''')
        my_bar.progress(30)

        if 'Response Time' in df.columns:
            if is_RT_sec:
                min_rt = str(round(df['Response Time'].min(), 3)).rjust(10)
                mean_rt = str(round(df['Response Time'].mean(), 3)).rjust(10)
                median_rt = str(round(df['Response Time'].median(), 3)).rjust(10)
                max_rt = str(round(df['Response Time'].max(), 3)).rjust(10)
                unit_rs = '秒'
            else:
                min_rt = str(int(df['Response Time'].min())).rjust(10)
                mean_rt = str(int(df['Response Time'].mean())).rjust(10)
                median_rt = str(int(df['Response Time'].median())).rjust(10)
                max_rt = str(int(df['Response Time'].max())).rjust(10)
                unit_rs = 'ミリ秒'
            st.markdown(f'''
            ### 応答時間の集計結果
            * 最小値 : {min_rt} {unit_rs}
            * 平均値 : {mean_rt} {unit_rs}
            * 中央値 : {median_rt} {unit_rs}
            * 最大値 : {max_rt} {unit_rs}
            ''')

            st.markdown('### 応答時間のワースト5')
            df_sorted = df.sort_values('Response Time', ascending=False).head(5)
            df_sorted
        my_bar.progress(40)

        if 'User Agent' in df.columns:
            st.markdown('### クライアントOSとユーザーエージェントの種類')
            ua_df = DataFrame(df.groupby(['User Agent']).size().index)
            ua_df['count'] = df.groupby(['User Agent']).size().values
            cnt = replace_df_minors_with_others(ua_df, 'count').reset_index(drop=True)
            cnt

            ua_counter = {}
            os_counter = {}

            for index, row in ua_df.sort_values(['count'], ascending=False).iterrows():

                ua = woothee.parse(row['User Agent'])

                uaKey = ua.get('name') + ' (' + ua.get('version') + ')'
                if not uaKey in ua_counter:
                    ua_counter[uaKey] = 0
                ua_counter[uaKey] = ua_counter[uaKey] + 1

                osKey = ua.get('os') + ' (' + ua.get('os_version') + ')'
                if not osKey in os_counter:
                    os_counter[osKey] = 0
                os_counter[osKey] = os_counter[osKey] + 1

            fig = plt.figure(figsize = (15, 10))
            plt.subplot(1,2,1)
            plt.title('Client OS')
            os_counter_with_others = replace_dict_minors_with_others(os_counter)
            plt.pie(os_counter_with_others.values(), labels = os_counter_with_others.keys(), autopct = '%1.1f%%', startangle = 90)

            plt.subplot(1,2,2)
            plt.title('User Agent')
            ua_counter_with_others = replace_dict_minors_with_others(ua_counter)
            plt.pie(ua_counter_with_others.values(), labels = ua_counter_with_others.keys(), autopct = '%1.1f%%', startangle = 90)
            #plt.show()
            fig

            st.markdown('#### ユーザーエージェントの種類（全て）')
            ua_df_sorted = ua_df.sort_values(['count'], ascending=False)
            ua_df_sorted
        my_bar.progress(50)

        if 'Remote Host' in df.columns:
            st.markdown('### アクセス数の多いクライアント（リモートホスト）')
            rh_df = DataFrame(df.groupby(['Remote Host']).size().index)
            rh_df['count'] = df.groupby(['Remote Host']).size().values
            cnt = replace_df_minors_with_others(rh_df, 'count').reset_index(drop=True)
            cnt[["Country","City","lat","lon"]] = cnt.apply(lambda x:get_country(x),axis=1, result_type='expand')
            cnt
            st.map(cnt.dropna(how='any'))
        my_bar.progress(60)

        if 'Status' in df.columns:

            st.markdown('### レスポンスのステータスコード')
            status_df = DataFrame(df.groupby(['Status']).size().index)
            status_df['count'] = df.groupby(['Status']).size().values
            status_df['percentage'] = (df.groupby(['Status']).size() / len(df) * 100).values
            col1, col2 = st.columns(2)
            with col1:
                fig = plt.figure()
                labels = [str(n)+'xx' for n in list(df.groupby([df['Status'] // 100]).groups.keys())]
                plt.pie(df.groupby([df['Status'] // 100]).size(), autopct = '%1.1f%%', labels = labels, startangle = 90)
                plt.axis('equal')
                fig
            with col2:
                status_df

            st.markdown('### エラーレスポンスのステータスコード')
            error_df = status_df[status_df['Status'] >= 400]
            error_df['percentage'] = (error_df['count'] / error_df['count'].sum() * 100).values
            col1, col2 = st.columns(2)
            with col1:
                fig = plt.figure()
                labels = [str(int(n))+'xx' for n in list(df.groupby([error_df['Status'] // 100]).groups.keys())]
                plt.pie(error_df.groupby([error_df['Status'] // 100]).sum().iloc[:, 1], labels=labels, counterclock=False, startangle=90)
                labels2 = [str(n) for n in list(error_df['Status'].unique())]
                # 円グラフ (内側, 半径 70% で描画)
                plt.pie(error_df.iloc[:, 1], labels=labels2, counterclock=False, startangle=90, radius=0.7)
                # 中心 (0,0) に 40% の大きさで円を描画
                centre_circle = plt.Circle((0,0),0.4, fc='white')
                fig = plt.gcf()
                fig.gca().add_artist(centre_circle)
                fig
            with col2:
                error_df

            st.markdown('### エラーリクエストの詳細')
            error_df = df[df['Status'] >= 400]
            error_df
            
        my_bar.progress(70)

        if 'Request' in df.columns and 'Time' in df.columns:
            st.markdown('### 負荷の状況')
            fig = plt.figure(figsize = (15, 5))
            access = df['Request']
            access.index = df['Time']
            access = access.resample('S').count()
            access.index.name = 'Time'
            access.plot()
            plt.title('Total Access')
            plt.ylabel('Access')
            fig

        if 'Request' in df.columns and 'Time' in df.columns and 'Status' in df.columns:
            st.markdown('### エラーの発生状況')
            fig = plt.figure(figsize = (15, 5))
            error_access = df[['Request', 'Status']]
            error_access.index = df['Time']
            error_access = error_access.resample('S')['Status'].apply(lambda x: (x >= 400).sum())
            error_access.index.name = 'Time'
            error_access.plot()
            plt.title('Total Error Access')
            plt.ylabel('Error Access')
            ax = plt.gca()
            ax.yaxis.set_major_locator(MultipleLocator(1))
            fig
 
        if 'Request' in df.columns and 'Time' in df.columns and 'Status' in df.columns:
            st.markdown('### システムエラー（HTTP 5xx）の発生状況')
            fig = plt.figure(figsize = (15, 5))
            error_access = df[['Request', 'Status']]
            error_access.index = df['Time']
            error_access = error_access.resample('S')['Status'].apply(lambda x: (x >= 500).sum())
            error_access.index.name = 'Time'
            error_access.plot()
            plt.title('Total Error Access')
            plt.ylabel('Error Access')
            ax = plt.gca()
            ax.yaxis.set_major_locator(MultipleLocator(1))
            fig
 
        if 'Request' in df.columns and 'Time' in df.columns and 'Response Time' in df.columns:
            drow_delayed_access(df, is_RT_sec, 1)
            drow_delayed_access(df, is_RT_sec, 3)
            drow_delayed_access(df, is_RT_sec, 10)
            drow_delayed_access(df, is_RT_sec, 30)

        if 'Request' in df.columns:
            x = df['Request'].str.split(expand=True)
            x.columns = ['Method', 'URL', 'Version']
            titles = ['リクエストのメソッド', 'リクエストのURL', 'リクエストのバージョン']
            for idx, col_name in enumerate(x.columns):
                drow_pie_and_table(x, col_name, titles[idx])

        my_bar.progress(90)

        if 'Size' in df.columns and 'Response Time' in df.columns:
            st.markdown('### レスポンスのサイズと時間の関係性')
            fig = plt.figure(figsize = (15, 5))
            sec_unit = 1000
            if is_RT_sec:
                sec_unit = 1
            plt.scatter(df['Size']/1000, df['Response Time']/sec_unit, marker='.')
            plt.xlabel('Size (KB)')
            plt.ylabel('Response Time (sec)')
            plt.grid()
            fig
        my_bar.progress(100)
