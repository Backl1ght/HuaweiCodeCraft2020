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
    "141373": "京津东赛区",
    "141377": "上合赛区",
    "141374": "杭厦赛区",
    "141378": "江山赛区",
    "141375": "成渝赛区",
    "141379": "西北赛区",
    "141376": "武长赛区",
    "141380": "粤港澳赛区",
    "141381": "海外赛区",
    "141372": "复活赛区"
}


def show_rank(TeamList):
    f = open('./' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.csv', 'w', encoding='utf-8')
    TeamList.sort()
    rk = 1
    f.write("Rank,Stage,Name,Score,UserList\n")
    for team in TeamList:
        f.write("{},{},\"{}\",`{},".format(rk, dic[team.stageId], team.teamName, team.Score))
        for user in team.userList:
            f.write(user + '; ')
        f.write('\n')
        rk = rk + 1


if __name__ == '__main__':
    URLS = [
        'https://competition.huaweicloud.com/competition/v1/competitions/ranking/1000036574?stage_id=141373&page_no={}&page_size=10&_=1588831084072',
        'https://competition.huaweicloud.com/competition/v1/competitions/ranking/1000036576?stage_id=141377&page_no={}&page_size=10&_=1588831145005',
        'https://competition.huaweicloud.com/competition/v1/competitions/ranking/1000036577?stage_id=141374&page_no={}&page_size=10&_=1588831178250',
        'https://competition.huaweicloud.com/competition/v1/competitions/ranking/1000036578?stage_id=141378&page_no={}&page_size=10&_=1588831214626',
        'https://competition.huaweicloud.com/competition/v1/competitions/ranking/1000036579?stage_id=141375&page_no={}&page_size=10&_=1588831248741',
        'https://competition.huaweicloud.com/competition/v1/competitions/ranking/1000036580?stage_id=141379&page_no={}&page_size=10&_=1588831270845',
        'https://competition.huaweicloud.com/competition/v1/competitions/ranking/1000036581?stage_id=141376&page_no={}&page_size=10&_=1588831306105',
        'https://competition.huaweicloud.com/competition/v1/competitions/ranking/1000036582?stage_id=141380&page_no={}&page_size=10&_=1588831334580',
        'https://competition.huaweicloud.com/competition/v1/competitions/ranking/1000036583?stage_id=141381&page_no={}&page_size=10&_=1588831365728',
        'https://competition.huaweicloud.com/competition/v1/competitions/ranking/1000041211?stage_id=141372&page_no={}&page_size=10&_=1588831406374',
    ]

    TeamList = []

    for URL in URLS:
        get_content(URL, TeamList)

    show_rank(TeamList)

    print('Task finished!')
