from SoccerNet.Downloader import SoccerNetDownloader
mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="SoccerNetData")
#mySoccerNetDownloader.downloadDataTask(task="tracking", split=["train","test","challenge"])
mySoccerNetDownloader.downloadDataTask(task="tracking-2023", split=["train", "test", "challenge"])

#The Tracking dataset consists of 12 complete soccer games from the main camera including:

#200 clips of 30 seconds with tracking data.
#one complete halftime annotated with tracking data.
#the complete videos for the 12 games.