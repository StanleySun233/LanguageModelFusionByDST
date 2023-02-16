import sqlite3
import time

import utils


class SqliteHelper:
    def __init__(self, path):
        self.connection = None
        self.path = path
        self.connection: sqlite3.connect

    def setConnection(self):
        try:
            self.connection = sqlite3.connect(self.path, check_same_thread=False)
            utils.tools.logFormat(utils.tools.INFO, '成功连接数据库')
        except:
            utils.tools.logFormat(utils.tools.WARN, '数据库连接失败')
            exit(0)

    def insertInfo(self, table, args):
        begTime = time.time()
        cur = self.connection.cursor()
        if type(args) == list:
            value = ''
            for i in args:
                value += ("\'{}\',".format(str(i)))
            value = '({})'.format(value[:-1])
            sqlString = 'insert into {} values {}'.format(table, value)
        else:
            att = ''
            for i in args.keys():
                att += '{} ,'.format(i)
            att = '({})'.format(att[:-1])

            value = ''
            for i in args.keys():
                value += '\'{}\','.format(args[i])
            value = '({})'.format(value[:-1])
            sqlString = 'insert into {}{} values {}'.format(table, att, value)
        cur.execute(sqlString)
        self.connection.commit()

    def delInfo(self, table, attrs: dict):
        begTime = time.time()
        sqlString = 'delete from {} where {} = \'{}\''.format(table, [i for i in attrs.keys()][0],
                                                              attrs[[i for i in attrs.keys()][0]])
        cur = self.connection.cursor()
        cur.execute(sqlString)
        self.connection.commit()
        utils.tools.logFormat(utils.tools.INFO, "耗时 {}s 刪除 {}".format(round(time.time() - begTime, 6), sqlString))

    def searchInfo(self, table, attrs=None, val=None, mult=False, isLike=False, order=False):
        begTime = time.time()
        if attrs is None:
            attrs = ''
        if val is None:
            val = []

        if isLike:
            lk = 'like'
        else:
            lk = '='

        sel = ''
        for i in val:
            sel += '{},'.format(i)

        if len(val):
            sel = '{}'.format(sel[:-1])
        else:
            sel = '*'

        if len(attrs) == 0:
            sqlString = 'select {} from {}'.format(sel, table)
        else:
            sqlString = ''
            for i in attrs:
                if isLike:
                    sqlString += ('{} {} \'%{}%\' and '.format(i, lk, attrs[i]))
                else:
                    sqlString += ('{} {} \'{}\' and '.format(i, lk, attrs[i]))
            sqlString = 'select {} from {} where {} '.format(sel, table, sqlString[:-4])
        if order:
            sqlString += ' order by ({}.{})'.format(table, order)
        cur = self.connection.cursor()
        cur.execute(sqlString)
        res = cur.fetchall()
        if not mult:
            res = res[0]
        utils.tools.logFormat(utils.tools.INFO, '耗时 {}s 查询 {}'.format(round(time.time() - begTime, 6), sqlString))
        return res

    def isExist(self, table, attrs=None):
        begTime = time.time()
        if attrs is None:
            attrs = ''
        if len(attrs) == 0:
            sqlString = 'select * from {}'.format(table)
        else:
            sqlString = ''
            for i in attrs:
                sqlString += ('{} = \'{}\' and '.format(i, attrs[i]))
            sqlString = 'select * from {} where {}'.format(table, sqlString[:-4])
        cur = self.connection.cursor()
        cur.execute(sqlString)
        res = cur.fetchall()
        utils.tools.logFormat(utils.tools.INFO, '耗时 {}s 查询 {}'.format(round(time.time() - begTime, 6), sqlString))
        if len(res) > 0:
            return True
        else:
            return False

    def update(self, table, attrs: dict, val: dict):
        begTime = time.time()
        att = ''
        for i in attrs:
            att += '{} = \'{}\' and'.format(i, attrs[i])
        att = att[:-3]

        valu = ''
        for i in val:
            valu += '{} = \'{}\' and'.format(i, val[i])
        valu = valu[:-3]

        sqlString = 'update {} set {} where {}'.format(table, valu, att)
        cur = self.connection.cursor()
        cur.execute(sqlString)
        self.connection.commit()
        utils.tools.logFormat(utils.tools.INFO, '耗时 {}s 修改 {}'.format(round(time.time() - begTime, 6), sqlString))

    def getColumns(self, table):
        begTime = time.time()
        sqlString = 'pragma table_info({})'.format(table)
        cur = self.connection.cursor()
        cur.execute(sqlString)
        res = cur.fetchall()
        col = []
        for i in res:
            col.append(i[1])
        utils.tools.logFormat(utils.tools.INFO, '耗时 {}s 查询表头 {}'.format(round(time.time() - begTime, 6), sqlString))
        return col

    def query(self, query, method='s'):
        begTime = time.time()
        cur = self.connection.cursor()
        cur.execute(query)
        if method == 'u':
            self.connection.commit()
            utils.tools.logFormat(utils.tools.INFO, '耗时 {}s 执行 {}'.format(round(time.time() - begTime, 6), query))
        else:
            res = cur.fetchall()
            utils.tools.logFormat(utils.tools.INFO, '耗时 {}s 执行 {}'.format(round(time.time() - begTime, 6), query))
            return res
