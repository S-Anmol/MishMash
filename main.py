from tkinter import *
import subprocess

t=Tk()
'''
E1 = variable 1
E2 = variable 2
E3 = Result
'''
def transfer():
    #Function to add
    E3.delete(0, END)   
    m=E1.get()  
    n=E2.get()     
    E3.delete(0,END)
    E3.insert(10,"Transaction Starting")   

    subprocess.run(['python', 'blink/blink_detect.py', m, n])


UpperFrame=Frame(t)                 
                                    
UpperFrame.pack()
MiddleFrame=Frame(t)
MiddleFrame.pack()                  
t.geometry("500x200")               
label1=Label(t,text="Amount",fg="blue")   #fg=foreground colour, basically font colour
label1.pack(side=LEFT)
E1=Entry(t,width=20)
E1.pack(side=LEFT,padx=10,pady=10)

E2=Entry(t,width=20)
E2.pack(side=RIGHT,padx=10,pady=10)   #padx, pady is the boundaru or the minimum distance to the nearest object
label2=Label(t,text="Beneficiary",fg="blue")
label2.pack(side=RIGHT)
button1=Button(UpperFrame,text="Transfer",command=transfer,fg="red")  #All for buttons command= the function it calls, note no () after function
button1.pack(fill=X,pady=2)
E3=Entry(MiddleFrame,width=20)
E3.pack(side=LEFT,fill=X,padx=10,pady=10)
E3.insert(10,"Result")
t.mainloop()