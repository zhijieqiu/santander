# -*- coding: utf-8 -*-
def print_params(*params):
    print params
print_params("test")
print_params(1,2,3)

def test():
    yield 1
    yield 2
    yield 33
    yield 44
a =  test()
v = a.next()
print v
v = a.next()
print v
v = a.next()
print v
v = a.next()
print v
lst = range(3)
for  k  in lst:
    print k
it = iter(lst)
try:
    while True:
        val = it.next()
        print val
except StopIteration:
    pass
for idx,ele in enumerate(lst):
    print idx,ele
def story(**kwds):
    return 'once upon a tim. there was a %(job)s calls %(name)s'%kwds
def power(x,y,*others):
    if others:
        print 'received redudant parameters:',others
    return pow(x,y)
def interval(start,stop=None,step = 1):
    'Initates range() for stop > 0'
    if stop is None:
        start,stop =  0,start
    result =  []
    i = start
    while i < stop:
        result.append(i)
        i+=step
    return result
print story(job='king',name = 'Gumby')
print story(job='jack',name = 'qiuzhijie')
t = {'job' :'帅哥','name':'丘志杰'}
print story(**t)
print power(3,2)
print power(3,2,1,0)
power(*interval(3,7))

class Person:
    def __inaccessMethod(self):
        print 'I bet that you can not see me'
    def setName(self,name):
        self.name = name
    def getName(self):
        return  self.name
    def greeting(self):
        print 'hello,my name is ',  self.name
person = Person()
person.setName('jack')
person.greeting()
person._Person__inaccessMethod()




class Clock:
    def time(self,t):
        print 'the time is ',t
class SpeakClock(Clock):
    def time(self,t):
        print "I can speak,hahaha"
        print 'the time is ',t
sc = SpeakClock()
sc.time('18/12/2015')
try:
    x = input('please input x')
    y = input('please input y')
    print x/y
except ZeroDevisionException:
    print'the second number can not be zero'
else:
    print 'else is called here'
class Bird:
    def __init__(self):
        self.hungry = True
    def eat(self):
        if self.hungry:
            self.hungry = False
            print 'Aaaah....'
        else:
            print 'No,Thanks'
b = Bird()
b.eat()
b.eat()
class SongBird(Bird):
    def __init__(self):
        self.song = 'Squawk!'
    def sing(self):
        print  self.song
sb =  SongBird()
sb.sing()
class MyBird(Bird):
    def __init__(self):
        Bird.__init__(self)
        self.name =  'jack'
    def introduce(self):
        print 'my name is :',self.name
mb =MyBird()
mb.eat()
mb.eat()

def flatten(nested):
    try:
        for sublist in  nested:
            for element in flatten(sublist):
                yield element
    except TypeError:
        yield nested
nested = [[1,2],[3,4],[5,7]]
for ele in flatten(nested):
    print ele,','
def conflict(state,nextX):
    nextY =  len(state)
    for i in range(nextY):
        if abs(state[i]-nextX) in (0,nextY-i):
            return True
    return False


def queens(num=8,state=()):
    for pos in range(num):
        if not conflict(state,pos):
            if len(state)==num-1:
                yield (pos,)
            else:
                for result in queens(num,state+(pos,)):
                    yield (pos,)+result

for result in queens():
    print result

def prettyprint(solution):
    def line(pos,length = len(solution)):
        return '. '*(pos)+'X'+'. '*(length-pos-1)
    for pos in solution:
        print line(pos)

print len(list(queens()))


