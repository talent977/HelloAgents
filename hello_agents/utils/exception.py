#!/usr/bin/env python
# -*- coding: utf-8 -*-


class Resp:
    @staticmethod
    def success(self, msg='',code=200, result=None):
        return {'code': code, 'msg': msg, 'result': result}

    @staticmethod
    def fail(self, msg='',code=-1, result=None):
        return {'code': code, 'msg': msg, 'result': result}