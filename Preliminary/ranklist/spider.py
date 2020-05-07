import requests
import json
import time


class Team:
    def __init__(self, teamName, Score, stageId, userlist):
        self.teamName = teamName
        self.Score = Score
        self.stageId = stageId
        self.userList = userlist

    def __lt__(self, other):
        return self.Score < other.Score


def get_content(_URL, TeamList):
    URL = _URL.format(1)

    html = requests.get(URL)
    json_obj = json.loads(html.text)

    t = json.dumps(json_obj)
    print(t)

    result = json_obj['result']
    teamRankingList = result['teamRankingList']
    totalPage = teamRankingList['totalPage']

    for i in range(totalPage):
        URL = _URL.format(i + 1)
        html = requests.get(URL)
        json_obj = json.loads(html.text)
        result = json_obj['result']
        ranklist = result['teamRankingList']
        results = ranklist['results']
        for item in results:
            score = item['score']
            teamname = item['teamName']
            stageId = item['stageId']
            userList = []
            ulist = item['userList']
            for user in ulist:
                username = user['domainName']
                userList.append(username)
            TeamList.append(Team(teamName=teamname, Score=score, stageId=stageId, userlist=userList))


dic = {
    "136710": "京津东赛区",
    "136712": "上合赛区",
    "136713": "杭厦赛区",
    "136714": "江山赛区",
    "136715": "成渝赛区",
    "136716": "西北赛区",
    "136717": "武长赛区",
    "136718": "粤港澳赛区",
    "136719": "海外赛区"
}


def show_rank(TeamList):
    f = open('./' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.csv', 'w', encoding='utf-8')
    TeamList.sort()
    rk = 1
    f.write("Rank,Stage,Name,Score,UserList\n")
    for team in TeamList:
        f.write("{},{},{},'{},".format(rk, dic[team.stageId], team.teamName, team.Score))
        for user in team.userList:
            f.write(user + '; ')
        f.write('\n')
        rk = rk + 1


if __name__ == '__main__':
    URLS = [
        'https://competition.huaweicloud.com/competition/v1/competitions/ranking/1000036574?stage_id=136710&page_no={}&page_size=10&_=1586420004473',
        'https://competition.huaweicloud.com/competition/v1/competitions/ranking/1000036576?stage_id=136712&page_no={}&page_size=10&_=1586420093026',
        'https://competition.huaweicloud.com/competition/v1/competitions/ranking/1000036577?stage_id=136713&page_no={}&page_size=10&_=1586420126533',
        'https://competition.huaweicloud.com/competition/v1/competitions/ranking/1000036578?stage_id=136714&page_no={}&page_size=10&_=1586420176619',
        'https://competition.huaweicloud.com/competition/v1/competitions/ranking/1000036579?stage_id=136715&page_no={}&page_size=10&_=1586420214291',
        'https://competition.huaweicloud.com/competition/v1/competitions/ranking/1000036580?stage_id=136716&page_no={}&page_size=10&_=1586420240298',
        'https://competition.huaweicloud.com/competition/v1/competitions/ranking/1000036581?stage_id=136717&page_no={}&page_size=10&_=1586418561747',
        'https://competition.huaweicloud.com/competition/v1/competitions/ranking/1000036582?stage_id=136718&page_no={}&page_size=10&_=1586420260764',
        'https://competition.huaweicloud.com/competition/v1/competitions/ranking/1000036583?stage_id=136719&page_no={}&page_size=10&_=1586420294132',
    ]

    TeamList = []

    for URL in URLS:
        get_content(URL, TeamList)

    show_rank(TeamList)

    print('Task finished!')
