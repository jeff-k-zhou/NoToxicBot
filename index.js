const { Client, GatewayIntentBits, Events, Permissions, PermissionFlagsBits } = require("discord.js");
const axios = require("axios");
require("dotenv").config()
const util = require("util")

const client = new Client({
	intents: [
		GatewayIntentBits.Guilds,
		GatewayIntentBits.MessageContent,
		GatewayIntentBits.GuildMessages,
		GatewayIntentBits.GuildMembers
	]
});

client.once(Events.ClientReady, readyClient => {
	console.log(`Ready! Logged in as ${readyClient.user.tag}`);
})

client.login(process.env.TOKEN);

client.on(Events.MessageCreate, async (message) => {
	console.log(message.author.bot)
	if (!message.author.bot) {
		const result = await axios.post(`http://127.0.0.1:5000/predict`, {
			"text": message.content
		})
		console.log("done")
		if (result.data[0].label === "toxic") {
			message.delete()
			message.guild.members.fetch(message.author.id).then(member => {
				if (!member.permissions.has(PermissionFlagsBits.MuteMembers)) {
					member.timeout(3600000, "Toxicity detected in message").then(() => {
						message.channel.send(`<@${message.author.id}>, you have been muted for 1 hour due to toxicity in your message`)
					})
				}
			})
		}
	}
})