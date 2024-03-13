import time

class TimeRecord:
    def __init__(self, round=2):
        self.__start_time = 0
        self.__time_list = []
        self.__round = round

    def start(self):
        self.__start_time = time.time()

    def end(self, is_show=False):
        __end_time = time.time()
        __elapsed_time = __end_time - self.__start_time
        self.__time_list.append(__elapsed_time)
        if is_show:
            print(f"걸린 시간: {__elapsed_time}s")

    def get_time_list(self):
        return self.__time_list
    
    def get_max(self):
        return max(self.__time_list)
    
    def get_min(self):
        return min(self.__time_list)
    


    def show_avg(self):
        result = round(sum(self.__time_list) / len(self.__time_list), self.__round)
        print(f"{len(self.__time_list)}개의 리스트 개당 평균 시간: {result}s")

    def show_time_list(self):
        print(f"{self.__time_list}") 

    def show_max(self):
        print(f"리스트 내 최대치: {max(self.__time_list)}")

    def show_min(self):
        print(f"리스트 내 최소치: {min(self.__time_list)}")


    